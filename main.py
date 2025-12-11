import os
import h5py
import torch
import config
import numpy as np
import train_ori
import scipy.io as sio
from utils import data_preprocess
import pandas as pd
import scanpy as sc
import re

# ---- Matplotlib: 先设置后端，再导入 pyplot（很重要）----
import matplotlib

matplotlib.use("Agg")  # 无显示界面也能保存图形
import matplotlib.pyplot as plt

# 设置全局参数，优化矢量图输出
plt.rcParams['pdf.fonttype'] = 42  # 使用 TrueType 字体
plt.rcParams['ps.fonttype'] = 42  # PostScript 中使用 TrueType 字体
plt.rcParams['svg.fonttype'] = 'none'  # SVG 中保持文字为文字而非路径

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on device: {device}")


# === t-SNE & UMAP plot ===
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _get_labels_for_scatter(_res, fallback_true=None):
    """优先用预测标签；没有就退回真标签"""
    if isinstance(_res, dict) and "predicted_labels" in _res and _res["predicted_labels"] is not None:
        return np.asarray(_res["predicted_labels"]).reshape(-1)
    if fallback_true is not None:
        return np.asarray(fallback_true).reshape(-1)
    return None


def _plot_scatter_2d(X2d, labels, title, save_path, vector_format='both'):
    """
    绘制散点图并保存为矢量图
    vector_format: 'svg', 'pdf', 'both' (同时保存两种格式)
    """
    plt.figure(figsize=(6, 5), dpi=180)
    if labels is None:
        plt.scatter(X2d[:, 0], X2d[:, 1], s=6, alpha=0.8, rasterized=False)
    else:
        # 给每个簇一种颜色
        labels = labels.astype(int)
        for lb in np.unique(labels):
            m = labels == lb
            # rasterized=False 确保散点保持矢量格式
            plt.scatter(X2d[m, 0], X2d[m, 1], s=6, alpha=0.85, label=str(lb), rasterized=False)
        if np.unique(labels).size <= 20:
            plt.legend(loc="best", fontsize=7, markerscale=2, frameon=False)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()

    # 根据设置保存不同格式
    base_path = os.path.splitext(save_path)[0]
    if vector_format in ['svg', 'both']:
        svg_path = base_path + '.svg'
        plt.savefig(svg_path, bbox_inches="tight", format='svg')
        print(f"[viz] saved (SVG): {svg_path}")
    if vector_format in ['pdf', 'both']:
        pdf_path = base_path + '.pdf'
        plt.savefig(pdf_path, bbox_inches="tight", format='pdf')
        print(f"[viz] saved (PDF): {pdf_path}")
    # 可选：同时保存 PNG 作为预览
    if vector_format == 'both':
        png_path = base_path + '.png'
        plt.savefig(png_path, bbox_inches="tight", format='png', dpi=300)
        print(f"[viz] saved (PNG preview): {png_path}")

    plt.close()


# ===== 全局：持久化同一场运行中的 PCA+UMAP 映射器 =====
_UMAP_REDUCER = None  # umap.UMAP 实例（fit 一次，后续 transform）
_UMAP_PCA = None  # sklearn.decomposition.PCA 实例（可选，用于加速且保持一致）
_UMAP_FIT_TAG = None  # 记录是在哪个 tag（如 sub_ep100）上 fit 的
_UMAP_REF_EMB = None  # 参考 epoch（fit 时）的 2D 坐标，用来做 Procrustes 对齐
_UMAP_REF_STAT = None  # (mean, std)，用于简单标准化辅助对齐

_VIZ_DONE = set()  # 已处理过的 tag（避免一个 epoch 画多次）
_UMAP_BUFFER = {}  # 在参考 epoch 之前先缓存起来：tag -> (res_dict, true_labels)


def _parse_epoch_from_tag(tag: str) -> int:
    m = re.search(r'_ep(\d+)$', tag)
    return int(m.group(1)) if m else -1


def _ensure_numeric_dense(X):
    import numpy as _np
    try:
        import scipy.sparse as _sp
    except Exception:
        _sp = None

    # torch -> numpy
    if "torch" in str(type(X)):
        try:
            X = X.detach().cpu().numpy()
        except Exception:
            X = _np.array(X)

    if _sp is not None and _sp.issparse(X):
        X = X.A
    X = _np.asarray(X)
    if not _np.issubdtype(X.dtype, _np.number):
        X = X.astype(_np.float32)
    if _np.isnan(X).any() or _np.isinf(X).any():
        X = _np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    return X


def _procrustes_align_to_ref(X, ref):
    """将 X 对齐（旋转/缩放/平移）到 ref，返回对齐后的坐标。
       要求 X.shape == ref.shape == (N, 2)，且同一顺序的点是一一对应的同一细胞。"""
    import numpy as _np
    X = _np.asarray(X, dtype=_np.float64)
    Y = _np.asarray(ref, dtype=_np.float64)
    # 去均值
    X0 = X - X.mean(axis=0, keepdims=True)
    Y0 = Y - Y.mean(axis=0, keepdims=True)
    # 归一化尺度（避免大小差异导致数值不稳）
    normX = _np.linalg.norm(X0)
    normY = _np.linalg.norm(Y0)
    if normX < 1e-12 or normY < 1e-12:
        return X  # 退化情况直接返回
    X0 /= normX
    Y0 /= normY
    # SVD 求最优旋转
    U, _, VT = _np.linalg.svd(X0.T @ Y0)
    R = U @ VT
    # 最优统一缩放
    s = (X0 @ R * Y0).sum() / (X0 * X0).sum()
    # 应用变换并加回参考均值
    X_aligned = s * (X - X.mean(axis=0, keepdims=True)) @ R + Y.mean(axis=0, keepdims=True)
    return X_aligned


def plot_tsne_umap_from_results(res, true_labels=None, dataset_name="dataset",
                                use_pca_for_umap=True, pca_dim=50,
                                umap_params=None, fit_strategy="first_call",
                                vector_format='both'):
    """
    固定 UMAP 映射器版本，输出矢量图：
    - 第一次调用时：拟合（可选先 PCA 到 pca_dim，再 UMAP.fit）
    - 后续调用：用同一个 PCA+UMAP 对新特征 transform，从而坐标对齐

    vector_format: 'svg', 'pdf', 'both' (同时保存两种格式)
    """
    global _UMAP_REDUCER, _UMAP_PCA, _UMAP_FIT_TAG

    # 0) 取嵌入 & 标签
    if not isinstance(res, dict) or ("features" not in res) or res["features"] is None:
        print("[viz] WARN: results 中无 'features'，t-SNE/UMAP 跳过。")
        return
    X = _ensure_numeric_dense(res["features"])

    labels = _get_labels_for_scatter(res, fallback_true=true_labels)
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] != X.shape[0]:
            print("[viz] WARN: labels 数量与特征不一致，忽略标签着色。")
            labels = None

    out_dir = _ensure_dir(os.path.join("plot", "embed"))
    prefix = f"{dataset_name}"

    # 1) t-SNE（保持原有做法：每次各自跑，不强行对齐）
    from sklearn.manifold import TSNE
    lr = max(50.0, min(1000.0, X.shape[0] / 12.0))
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, X.shape[0] // 50)),
        learning_rate=lr,
        init="pca",
        metric="cosine",
        random_state=0,
        n_iter=1500,
        n_iter_without_progress=300,
        verbose=1
    )
    X_tsne = tsne.fit_transform(X)
    _plot_scatter_2d(X_tsne, labels, title=f"{prefix} - tSNE",
                     save_path=os.path.join(out_dir, f"{prefix}_tSNE"),
                     vector_format=vector_format)

    # 2) UMAP（固定映射器 + Procrustes 对齐）
    import umap
    from sklearn.decomposition import PCA

    default_umap = dict(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=0, n_components=2, n_epochs=None)
    if umap_params:
        default_umap.update(umap_params)

    need_fit = False
    if _UMAP_REDUCER is None:
        if fit_strategy == "first_call":
            need_fit = True
        elif fit_strategy == "ref_epoch":
            ref_tag = os.environ.get("UMAP_REF_TAG", "")
            need_fit = (ref_tag != "") and (dataset_name == ref_tag)

    # 统一输入维度（可选 PCA）
    if use_pca_for_umap:
        if (_UMAP_PCA is None) and need_fit:
            _UMAP_PCA = PCA(n_components=min(pca_dim, X.shape[1]), random_state=0).fit(X)
        X_umap_in = _UMAP_PCA.transform(X) if _UMAP_PCA is not None else PCA(
            n_components=min(pca_dim, X.shape[1]), random_state=0).fit_transform(X)
    else:
        X_umap_in = X

    global _UMAP_REF_EMB, _UMAP_REF_STAT

    if need_fit:
        _UMAP_REDUCER = umap.UMAP(**default_umap)
        X_umap = _UMAP_REDUCER.fit_transform(X_umap_in)
        _UMAP_FIT_TAG = dataset_name
        # 记录参考嵌入及其简单统计量（用于后续对齐）
        _UMAP_REF_EMB = X_umap.copy()
        _UMAP_REF_STAT = (X_umap.mean(axis=0), X_umap.std(axis=0) + 1e-8)
        print(f"[viz] UMAP fitted on: {_UMAP_FIT_TAG} (use_pca={use_pca_for_umap}, dim_in={X_umap_in.shape[1]})")
    else:
        if _UMAP_REDUCER is None:
            # 安全兜底（理论上走不到）
            _UMAP_REDUCER = umap.UMAP(**default_umap)
            X_umap = _UMAP_REDUCER.fit_transform(X_umap_in)
            _UMAP_FIT_TAG = dataset_name
            _UMAP_REF_EMB = X_umap.copy()
            _UMAP_REF_STAT = (X_umap.mean(axis=0), X_umap.std(axis=0) + 1e-8)
            print(f"[viz] UMAP fitted (fallback) on: {_UMAP_FIT_TAG}")
        else:
            if use_pca_for_umap and (_UMAP_PCA is not None):
                X_umap_in = _UMAP_PCA.transform(X)
            X_umap = _UMAP_REDUCER.transform(X_umap_in)

            # --- 对齐步骤：先按参考均值/方差做简单标准化，再做正交 Procrustes ---
            if _UMAP_REF_EMB is not None:
                mu_ref, std_ref = _UMAP_REF_STAT
                mu = X_umap.mean(axis=0)
                std = X_umap.std(axis=0) + 1e-8
                X_std = (X_umap - mu) / std * std_ref + mu_ref  # 粗对齐
                X_umap = _procrustes_align_to_ref(X_std, _UMAP_REF_EMB)  # 精对齐

    _plot_scatter_2d(X_umap, labels, title=f"{prefix} - UMAP (aligned via {_UMAP_FIT_TAG})",
                     save_path=os.path.join(out_dir, f"{prefix}_UMAP_aligned"),
                     vector_format=vector_format)


def viz_callback(res_dict, true_labels, tag: str, vector_format='both'):
    global _VIZ_DONE, _UMAP_BUFFER, _UMAP_REDUCER

    # 同一 tag 只画一次（train_ori 里会在同一 epoch 的多个 mini-batch 调用）
    if tag in _VIZ_DONE:
        return

    ref_tag = os.environ.get("UMAP_REF_TAG", f"{config.args.name}_ep100")

    # 1) 还没拟合 UMAP 且这不是参考 epoch：先缓存，等参考 epoch 再统一出图
    if (_UMAP_REDUCER is None) and (tag != ref_tag):
        _UMAP_BUFFER[tag] = (res_dict, true_labels)
        _VIZ_DONE.add(tag)
        print(f"[viz] buffered {tag} (waiting for {ref_tag})")
        return

    # 2) 是参考 epoch 且尚未拟合：在这里 fit，然后把之前缓存的也一并 transform & 画图
    if (_UMAP_REDUCER is None) and (tag == ref_tag):
        # 参考 epoch：显式用 first_call 触发 fit
        plot_tsne_umap_from_results(
            res_dict, true_labels=true_labels, dataset_name=tag,
            fit_strategy="first_call", vector_format=vector_format
        )
        _VIZ_DONE.add(tag)
        # 把缓存的早期 epoch 按顺序刷出来（用 ref_epoch 模式，只 transform）
        for t in sorted(_UMAP_BUFFER.keys(), key=_parse_epoch_from_tag):
            rd, tl = _UMAP_BUFFER[t]
            plot_tsne_umap_from_results(
                rd, true_labels=tl, dataset_name=t,
                fit_strategy="ref_epoch", vector_format=vector_format
            )
            _VIZ_DONE.add(t)
        _UMAP_BUFFER.clear()
        print(f"[viz] flushed buffered epochs aligned to {tag}")
        return

    # 3) UMAP 已拟合（>参考 epoch）：直接 transform & 对齐
    plot_tsne_umap_from_results(
        res_dict, true_labels=true_labels, dataset_name=tag,
        fit_strategy="ref_epoch", vector_format=vector_format
    )
    _VIZ_DONE.add(tag)


if __name__ == "__main__":
    # ===== 设定 UMAP 的参考 epoch 为 ep100（例如 "sub_ep100"）=====
    # 注意：必须在训练前设置
    os.environ["UMAP_REF_TAG"] = f"{config.args.name}_ep100"

    # 清空全局映射器（如需重跑）
    _UMAP_REDUCER = None
    _UMAP_PCA = None
    _UMAP_FIT_TAG = None

    h5_datasets = []
    mat_datasets = []

    current_dir = os.getcwd()
    selected_dataset = config.args.name
    gene_expr = []
    true_labels = []

    # 优先用 data_path 参数加载数据
    use_external_path = config.args.data_path and os.path.exists(config.args.data_path)
    file_path = config.args.data_path if use_external_path else os.path.join(current_dir, "data",
                                                                             f"{selected_dataset}.h5ad")

    # h5/h5mat 逻辑保持不变
    if selected_dataset in h5_datasets and not use_external_path:
        file_path = os.path.join(current_dir, "data", f"{selected_dataset}.h5")
        h5_file = h5py.File(file_path, 'r')
        gene_expr = np.array(h5_file.get('X'))
        true_labels = np.array(h5_file.get('Y')).reshape(-1)
        gene_expr = data_preprocess(gene_expr, config.args.select_gene)
    elif selected_dataset in mat_datasets and not use_external_path:
        file_path = os.path.join(current_dir, "data", f"{selected_dataset}.mat")
        mat_file = sio.loadmat(file_path)
        gene_expr = np.array(mat_file['feature'])
        true_labels = np.array(mat_file['label']).reshape(-1)
        gene_expr = data_preprocess(gene_expr, config.args.select_gene)
    else:
        # .h5ad 逻辑
        if not os.path.exists(file_path):
            file_path = r"F:\pytorch_learning\newmethod\trytry\data\merge.h5ad"
        adata = sc.read_h5ad(file_path)

        # 优先用 label_col 参数，否则用常用标签列
        if config.args.label_col and config.args.label_col in adata.obs:
            label_key = config.args.label_col
        else:
            possible_label_keys = ['cluster', 'Cluster', 'cell_type', 'CellType', 'type', 'label', 'WGCNAcluster',
                                   'Louvain Cluster']
            label_key = None
            for k in possible_label_keys:
                if k in adata.obs:
                    label_key = k
                    break

        if label_key is None:
            raise ValueError(
                f'没有在 .obs 里找到标签列（尝试过：{[config.args.label_col] + possible_label_keys}），实际可用列：{list(adata.obs.columns)}')

        # 只保留有label且不为"unsure"和非空的cell
        valid_mask = (~adata.obs[label_key].isna()) & (adata.obs[label_key] != 'unsure')
        adata = adata[valid_mask].copy()

        adata = data_preprocess(adata, config.args.select_gene)  # 预处理，假设返回 AnnData
        import scipy.sparse

        if scipy.sparse.issparse(adata.X):
            gene_expr = adata.X.toarray().astype('float32')
        else:
            gene_expr = adata.X.astype('float32')
        true_labels = adata.obs[label_key].values

        # 数字化标签
        label_encoder = {k: v for v, k in enumerate(pd.unique(true_labels))}
        true_labels = pd.Series(true_labels).map(label_encoder).values

        print("After preprocessing, gene_expr shape:", gene_expr.shape)
        print("true_labels.shape:", true_labels.shape)
        assert gene_expr.shape[0] == true_labels.shape[0], "标签和表达矩阵数量不一致！"

    # --------- 无论如何，最后都保险处理一次 -----------
    gene_expr = np.array(gene_expr)
    if isinstance(gene_expr, list) or not hasattr(gene_expr, "shape") or getattr(gene_expr, "dtype", None) == object:
        gene_expr = np.array(gene_expr, dtype=np.float32)
    print("gene_expr type:", type(gene_expr), "dtype:", getattr(gene_expr, 'dtype', None))
    print(f"Gene expression matrix dimensions: {gene_expr.shape}")

    num_clusters = np.unique(true_labels).shape[0]
    print(f"Detected number of clusters: {num_clusters}")

    # epoch 选择优先用 max_epoch，否则 fallback 用 epoch
    epochs = config.args.max_epoch if hasattr(config.args, "max_epoch") and config.args.max_epoch else config.args.epoch


    # 修改 viz_callback 以支持矢量图格式
    def viz_callback_wrapper(res_dict, true_labels, tag: str):
        return viz_callback(res_dict, true_labels, tag, vector_format='both')


    # 后续训练
    results = train_ori.run(
        gene_exp=gene_expr,
        cluster_number=num_clusters,
        dataset=config.args.name,
        real_label=true_labels,
        epochs=epochs,
        lr=config.args.lr,
        temperature=config.args.temperature,
        dropout=config.args.dropout,
        layers=[config.args.enc_1, config.args.enc_2, config.args.enc_3, config.args.mlp_dim],
        save_pred=True,
        cluster_methods=config.args.cluster_methods,
        batch_size=config.args.batch_size,
        m=config.args.m,
        noise=config.args.noise,
        lambd=config.args.lambd,
        beta=config.args.beta,
        viz_epochs=[25, 50, 75, 100],
        viz_callback=viz_callback_wrapper
    )

    # === [NEW] 持久化聚类结果与嵌入，用于下游分析 ===
    import os, numpy as np, pandas as pd

    # 1) 取出预测标签与特征（train_ori.run 已返回到 results）
    pred = np.asarray(results.get("predicted_labels")) if "predicted_labels" in results else None
    feats = np.asarray(results.get("features")) if "features" in results else None
    assert pred is not None and pred.ndim == 1, "没有拿到 predicted_labels，请确认 save_pred=True 且 KMeans 正常运行。"

    # 2) 输出目录（默认放到输入 h5ad 同级目录下的 sdmcc_out）
    out_dir = os.path.join(os.path.dirname(file_path), "sdmcc_out")
    os.makedirs(out_dir, exist_ok=True)

    # 3) 保存 CSV（给 R 直接 join 到 meta 用）
    cells = pd.Index(adata.obs_names).astype(str) if 'adata' in globals() and adata is not None else pd.Index(
        range(pred.shape[0])).astype(str)
    pd.DataFrame({"cell": cells, "SDMCC_pred": pred.astype(int)}).to_csv(
        os.path.join(out_dir, f"{config.args.name}_SDMCC_pred.csv"), index=False)

    if feats is not None and feats.ndim == 2:
        df_emb = pd.DataFrame(feats, index=cells, columns=[f"z{i + 1}" for i in range(feats.shape[1])])
        df_emb.to_csv(os.path.join(out_dir, f"{config.args.name}_SDMCC_emb.csv"))

    # 4) 保存带注释的 h5ad（取消原来被注释掉的逻辑）
    try:
        import anndata as ad
    except Exception as _e:
        ad = None
        print("[save] 警告：未安装 anndata，仅保存 CSV。pip install -i https://pypi.tuna.tsinghua.edu.cn/simple anndata")

    if 'adata' in globals() and adata is not None:
        # 与训练时同一顺序：main.py 前面对 adata 过滤/预处理后再取 X 训练，顺序一致
        adata.obs["SDMCC_pred"] = pd.Categorical(pred)
        if feats is not None and feats.ndim == 2:
            adata.obsm["X_SDMCC"] = feats.astype("float32")
    else:
        # 若没有现成 adata，就新建一个只带 obs/obsm 的 AnnData
        if ad is not None:
            X_stub = feats.astype("float32") if feats is not None else np.zeros((pred.shape[0], 1), dtype=np.float32)
            adata = ad.AnnData(X=X_stub)
            adata.obs_names = cells
            adata.obs["SDMCC_pred"] = pd.Categorical(pred)
            if feats is not None and feats.ndim == 2:
                adata.obsm["X_SDMCC"] = feats.astype("float32")

    # 记录指标到 .uns
    metric_keys = ["acc", "ari", "nmi", "f1", "purity", "homo", "comp", "v_measure", "n_true_cluster", "n_pred_cluster",
                   "time"]
    if 'adata' in globals() and adata is not None:
        if "SDMCC_metrics" not in adata.uns:
            adata.uns["SSDMCC_metrics"] = {}
        for k in metric_keys:
            if k in results:
                adata.uns["SDMCC_metrics"][k] = results[k]

        if ad is not None:
            out_h5ad = os.path.join(out_dir, f"{config.args.name}_SDMCC_annotated.h5ad")
            adata.write_h5ad(out_h5ad)
            print(f"[save] 已保存：\n  {out_h5ad}")

    print(f"[save] 已保存：\n  {os.path.join(out_dir, f'{config.args.name}_SDMCC_pred.csv')}")
    if feats is not None and feats.ndim == 2:
        print(f"  {os.path.join(out_dir, f'{config.args.name}_SDMCC_emb.csv')}")


    def safe_print(results, key, name=None):
        if name is None:
            name = key
        print(f"{name}: ", results[key] if key in results else "(N/A)")


    safe_print(results, "acc", "ACC")
    safe_print(results, "ari", "ARI")
    safe_print(results, "nmi", "NMI")
    safe_print(results, "f1", "F1")
    safe_print(results, "purity", "Purity")
    safe_print(results, "homo", "Homogeneity")
    safe_print(results, "comp", "Completeness")
    safe_print(results, "v_measure", "V-measure")
    safe_print(results, "n_true_cluster", "True clusters")
    safe_print(results, "n_pred_cluster", "Pred clusters")
    safe_print(results, "time", "Time")
    print(results)

    try:
        import anndata as ad
    except Exception as e:
        ad = None
        print("Warning: anndata 未安装，若 adata 不存在将无法新建 AnnData。", e)

    # 1) 取出预测与特征（兼容不同键名）
    pred_keys = ["predicted_labels", "pred_labels", "pred", "y_pred"]
    feat_keys = ["features", "embeddings", "repr", "X"]

    pred = None
    for k in pred_keys:
        if isinstance(results, dict) and k in results:
            pred = np.asarray(results[k])
            break

    feats = None
    for k in feat_keys:
        if isinstance(results, dict) and k in results:
            feats = np.asarray(results[k])
            break

    if pred is None:
        raise RuntimeError("results 里未找到预测标签，请检查 train_ori.run 返回的键名。"
                           f" 已尝试键：{pred_keys}")

    # 2) 统一数据类型与长度
    pred = pred.astype(int).reshape(-1)
    if "true_labels" in results and results["true_labels"] is not None:
        true_labels = np.asarray(results["true_labels"]).reshape(-1)

    n_cells = pred.shape[0]
    if gene_expr.shape[0] != n_cells:
        raise ValueError(f"基因表达行数（{gene_expr.shape[0]}）与预测标签长度（{n_cells}）不一致！")

    # 如果是 view，先复制成可写对象
    if getattr(adata, "is_view", False):
        adata = adata.copy()

    # 3) 拿到/构造 AnnData
    need_new_adata = ("adata" not in globals()) or (adata is None)
    if need_new_adata:
        if ad is None:
            raise RuntimeError("未检测到现成 adata，且 anndata 未安装，无法新建 AnnData。"
                               "请先 pip install anndata（可用清华源）。")
        # 尽量保留 dtype 与稀疏性（若你上游是 csr_matrix，可转为 CSR 再喂给 AnnData）
        adata = ad.AnnData(X=gene_expr)
        print("已新建 AnnData。")
    else:
        if adata.shape[0] != n_cells:
            raise ValueError(f"现有 adata 细胞数（{adata.shape[0]}）与预测标签长度（{n_cells}）不一致！")

    # 4) 写入 obs/obsm/uns
    adata.obs["SDMCC_pred"] = pd.Categorical(pred)
    if feats is not None:
        # 确保是 float32，避免 R 侧读写时不必要的精度膨胀
        adata.obsm["X_SDMCC"] = feats.astype("float32")
    else:
        print("提示：results 中未找到特征向量（features/embeddings），将仅写入 SDMCC_pred。")

    # 记录指标（有啥写啥）
    metric_keys = [
        "acc", "ari", "nmi", "f1", "purity", "homo", "comp", "v_measure",
        "n_true_cluster", "n_pred_cluster", "time"
    ]
    if "SDMCC_metrics" not in adata.uns:
        adata.uns["SDMCC_metrics"] = {}
    for k in metric_keys:
        if k in results:
            adata.uns["SDMCC_metrics"][k] = results[k]

    # 5) 保存（按数据集名命名更清楚）
    out_name = f"{getattr(config.args, 'name', 'dataset')}_SDMCC_annotated.h5ad"
    adata.write_h5ad(out_name)
    print(f"Saved {out_name}")


# === 聚类一致性矩阵绘图 ===
def _extract_pred_labels_from_results(_res):
    """尝试从 results 字典里取预测标签"""
    if _res is None:
        return None
    for k in ["predicted_labels", "pred_labels", "preds", "pred", "y_pred", "labels_pred", "cluster_pred"]:
        if k in _res and _res[k] is not None:
            arr = np.asarray(_res[k]).reshape(-1)
            if arr.size > 0:
                return arr
    return None


def _get_features_for_similarity(res, fallback_features):
    """
    优先从 results 里拿低维表示作为相似度的特征（更能体现方法效果），
    否则退回到 gene_expr（高维表达矩阵）。
    """
    cand_keys = ["embedding", "emb", "z", "Z", "H", "feat", "feats", "features"]
    if isinstance(res, dict):
        for k in cand_keys:
            if k in res and res[k] is not None:
                X = np.asarray(res[k])
                if X.ndim == 2 and X.shape[0] > 1:
                    return X
    # fallback: 用基因表达
    X = np.asarray(fallback_features)
    return X


def _plot_similarity(labels, features, acc=None, title="Clustering", save_path=None,
                     draw_boxes=True, vector_format='both'):
    """
    论文风：灰底+青色块+红框+坐标轴+颜色条
    使用"特征余弦相似度矩阵"，按聚类标签排序后展示。
    输出矢量图格式。
    """
    from matplotlib import ticker as mticker
    from matplotlib.colors import LinearSegmentedColormap

    labels = np.asarray(labels).reshape(-1)
    assert features.shape[0] == labels.size, "features 与 labels 数量不一致"

    # === 计算余弦相似度（行归一化后做点积） ===
    X = np.asarray(features, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X = X / denom
    S = np.clip(X @ X.T, -1.0, 1.0)

    # 对比度增强 + 归一到 [0,1]
    q1, q99 = np.quantile(S, [0.01, 0.99])
    S = np.clip(S, q1, q99)
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)

    # === 按标签排序 ===
    idx = np.argsort(labels)
    S = S[idx][:, idx]
    labels_sorted = labels[idx]
    n = labels.size

    # === 画图 ===
    fig = plt.figure(figsize=(5.6, 5.6), dpi=180)
    ax = plt.gca()

    # 自定义「灰 -> 青」渐变：低值灰、凸显高相似度为青
    cmap_gc = LinearSegmentedColormap.from_list("gray_to_cyan", ["#d9d9d9", "#00bcd4"])

    # 灰色画布（轴与空白区域）
    fig.patch.set_facecolor("#f5f5f5")
    ax.set_facecolor("#f5f5f5")

    # 用自定义配色渲染矩阵
    # 关键：设置 rasterized=False 保持矢量格式
    im = ax.imshow(
        S,
        cmap=cmap_gc,
        vmin=0.0, vmax=1.0,
        interpolation="nearest",
        aspect="equal",
        rasterized=False  # 保持矢量格式
    )

    # 轴刻度
    if n <= 12:
        tick_step = 1
    else:
        rough = max(1, int(np.ceil(n / 8)))
        base = 50 if n <= 1000 else 100
        tick_step = int(np.ceil(rough / base) * base)
    ticks = np.arange(0, n, max(1, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.tick_params(axis='both', which='both', length=0, labelsize=8)

    ax.set_xlabel("Cells (sorted by cluster)", fontsize=9)
    ax.set_ylabel("Cells (sorted by cluster)", fontsize=9)

    ttl = title
    if acc is not None:
        ttl += f" ({acc * 100:.2f}%)"
    ax.set_title(ttl, fontsize=11, pad=6)

    # 红色小方框（保持不变）
    if draw_boxes:
        uniq, counts = np.unique(labels_sorted, return_counts=True)
        start = 0
        for c in counts:
            end = start + int(c)
            xs = [start - 0.5, end - 0.5, end - 0.5, start - 0.5, start - 0.5]
            ys = [start - 0.5, start - 0.5, end - 0.5, end - 0.5, start - 0.5]
            ax.plot(xs, ys, color="red", linewidth=1.2)
            start = end

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout(pad=0.6)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        base_path = os.path.splitext(save_path)[0] if save_path else "similarity_plot"

        # 保存矢量图格式
        if vector_format in ['svg', 'both']:
            svg_path = f"{base_path}.svg"
            fig.savefig(svg_path, bbox_inches="tight", format='svg')
            print(f"[viz] saved (SVG): {svg_path}")
        if vector_format in ['pdf', 'both']:
            pdf_path = f"{base_path}.pdf"
            fig.savefig(pdf_path, bbox_inches="tight", format='pdf')
            print(f"[viz] saved (PDF): {pdf_path}")
        # 可选：同时保存高质量 PNG 作为预览
        if vector_format == 'both':
            png_path = f"{base_path}.png"
            fig.savefig(png_path, bbox_inches="tight", format='png', dpi=300)
            print(f"[viz] saved (PNG preview): {png_path}")

    plt.close(fig)


# === 这里开始是顶层代码（不是函数体里） ===
try:
    _labels_pred = _extract_pred_labels_from_results(results)
    if _labels_pred is not None:
        _tag = "SDMCC"
        if getattr(config.args, "ablate_contrast", False) or getattr(config.args, "ablate-contrast", False):
            _tag = "w.o Contrast"

        _acc = None
        for k in ["acc", "ACC"]:
            if k in results and results[k] is not None:
                try:
                    _acc = float(results[k])
                    break
                except Exception:
                    pass

        # 取特征：优先 results 里的 embedding，否则用 gene_expr
        _feats = _get_features_for_similarity(results, fallback_features=gene_expr)

        fig_dir = r"F:\pytorch_learning\newmethod\trytry\plot"
        os.makedirs(fig_dir, exist_ok=True)
        fig_name = f"{config.args.name}_{_tag.replace(' ', '_').replace('.', '')}"

        _plot_similarity(
            labels=_labels_pred,
            features=_feats,
            acc=_acc,
            title=_tag,
            save_path=os.path.join(fig_dir, fig_name),
            vector_format='both'  # 输出矢量图格式
        )
    else:
        print("[viz] WARN: results 中没找到预测标签，绘图跳过。")
except Exception as e:
    print(f"[viz] ERROR while plotting: {e}")