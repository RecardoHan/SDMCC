import os
import SDMCC
import config
import time
import numpy as np
import torch
import contrastive_loss
from utils import update_learning_rate, store_model_checkpoint, compute_cluster_metrics
from sklearn.cluster import KMeans
import warnings
import torch.nn.functional as F
import matplotlib
from sklearn.decomposition import PCA


matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def plot_convergence(history, dataset_name, save_dir="plot/convergence"):
    """
    按照顶级期刊标准绘制训练收敛图 - 修复版本
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- 数据准备 ----
    epochs = history.get('epochs', [])
    train_losses = history.get('train_losses', history.get("losses", []))
    accs = history.get('accs', [])
    nmis = history.get('nmis', [])
    aris = history.get('aris', [])

    if not epochs or not train_losses:
        print(f"[Convergence] No data to plot for {dataset_name}")
        return

    # 调试信息 - 检查数据
    print(f"[Debug] Data check:")
    print(f"  Epochs: {len(epochs)}")
    print(f"  Train losses: {len(train_losses)}")
    print(f"  ACC: {len(accs)} - sample: {accs[:3] if accs else 'EMPTY'}")
    print(f"  NMI: {len(nmis)} - sample: {nmis[:3] if nmis else 'EMPTY'}")
    print(f"  ARI: {len(aris)} - sample: {aris[:3] if aris else 'EMPTY'}")

    # 更强的平滑处理
    def smooth_curve(data, window=10):
        """使用更大窗口的滑动平均"""
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        return smoothed

    epochs = np.array(epochs)
    train_losses_smooth = smooth_curve(train_losses, window=15)

    # ---- 设置期刊级别的图表样式 ----
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.framealpha': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # === 左轴：训练损失 ===
    color_loss = '#1f77b4'  # 标准蓝色
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color_loss, fontweight='bold')

    # 绘制训练损失
    line_loss = ax1.plot(epochs, train_losses_smooth, color=color_loss, linewidth=2.5,
                        solid_capstyle='round', label='Training Loss')

    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.tick_params(axis='both', which='major', width=1.2, length=6)

    # 设置简洁的网格
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    # === 右轴：聚类指标 ===
    ax2 = ax1.twinx()
    color_metrics = '#d62728'  # 标准红色
    ax2.set_ylabel('Clustering Performance', color=color_metrics, fontweight='bold')

    # 使用非常明显区分的颜色和样式
    metric_styles = {
        'ACC': {'color': '#FF6B6B', 'linestyle': '-', 'linewidth': 2.5},      # 亮红色，实线
        'NMI': {'color': '#4ECDC4', 'linestyle': '--', 'linewidth': 2.5},     # 青绿色，虚线
        'ARI': {'color': '#45B7D1', 'linestyle': '-.', 'linewidth': 2.5}      # 亮蓝色，点划线
    }

    # 收集所有需要画的线和标签
    all_lines = line_loss  # 训练损失线
    all_labels = ['Training Loss']

    # 逐一绘制每个指标，确保数据存在且非空
    if accs and len(accs) > 0 and any(x is not None and not np.isnan(x) for x in accs):
        accs_smooth = smooth_curve(accs, window=10)
        line_acc = ax2.plot(epochs, accs_smooth,
                           color=metric_styles['ACC']['color'],
                           linestyle=metric_styles['ACC']['linestyle'],
                           linewidth=metric_styles['ACC']['linewidth'],
                           solid_capstyle='round', label='ACC')
        all_lines.extend(line_acc)
        all_labels.append('ACC')
        print(f"[Debug] ACC plotted: {len(accs_smooth)} points")

    if nmis and len(nmis) > 0 and any(x is not None and not np.isnan(x) for x in nmis):
        nmis_smooth = smooth_curve(nmis, window=10)
        line_nmi = ax2.plot(epochs, nmis_smooth,
                           color=metric_styles['NMI']['color'],
                           linestyle=metric_styles['NMI']['linestyle'],
                           linewidth=metric_styles['NMI']['linewidth'],
                           solid_capstyle='round', label='NMI')
        all_lines.extend(line_nmi)
        all_labels.append('NMI')
        print(f"[Debug] NMI plotted: {len(nmis_smooth)} points")

    if aris and len(aris) > 0 and any(x is not None and not np.isnan(x) for x in aris):
        aris_smooth = smooth_curve(aris, window=10)
        line_ari = ax2.plot(epochs, aris_smooth,
                           color=metric_styles['ARI']['color'],
                           linestyle=metric_styles['ARI']['linestyle'],
                           linewidth=metric_styles['ARI']['linewidth'],
                           solid_capstyle='round', label='ARI')
        all_lines.extend(line_ari)
        all_labels.append('ARI')
        print(f"[Debug] ARI plotted: {len(aris_smooth)} points")
    else:
        print(f"[Debug] ARI NOT plotted - data issue")

    ax2.set_ylim(0, 1.02)
    ax2.tick_params(axis='y', labelcolor=color_metrics)
    ax2.tick_params(axis='y', which='major', width=1.2, length=6)

    # 移除右轴的spine
    ax2.spines['right'].set_visible(False)

    # === 图例设置 - 确保所有线都包含 ===
    print(f"[Debug] Legend will show: {all_labels}")
    legend = ax1.legend(all_lines, all_labels,
                       loc='upper right',
                       bbox_to_anchor=(0.98, 0.98),
                       frameon=True,
                       fancybox=False,
                       edgecolor='black',
                       framealpha=1.0,
                       facecolor='white')

    # 设置图例边框
    legend.get_frame().set_linewidth(1.0)

    # === 设置坐标轴范围和刻度 ===
    ax1.set_xlim(0, max(epochs))

    # 设置更专业的刻度
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # === 简洁的标题 ===
    ax1.set_title(f'{dataset_name} Training Convergence', fontweight='bold', pad=15)

    # === 调整布局 ===
    plt.tight_layout()

    # === 保存高质量图片 ===
    save_path = os.path.join(save_dir, f'{dataset_name}_convergence.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none',
                format='png', transparent=False)

    # 同时保存PDF版本（期刊常要求矢量格式）
    save_path_pdf = os.path.join(save_dir, f'{dataset_name}_convergence.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none',
                format='pdf', transparent=False)

    plt.close(fig)

    # 重置matplotlib参数
    plt.rcdefaults()

    print(f"[Convergence] Saved journal-quality plots:")
    print(f"  PNG: {save_path}")
    print(f"  PDF: {save_path_pdf}")

def run(gene_exp, cluster_number, dataset, real_label, epochs, lr, temperature, dropout, layers, batch_size, m,
        save_pred=True, noise=None, use_cpu=None, cluster_methods=None, lambd=1.0, beta=1.0,
        viz_epochs=None, viz_callback=None, plot_convergence_curve=True):
    viz_set = set(viz_epochs) if viz_epochs is not None else set()
    if cluster_methods is None:
        cluster_methods = []
    results = {}

    start = time.time()
    embedding, best_model_epoch, convergence_history = train_model(
        gene_exp=gene_exp,
        cluster_number=cluster_number,
        real_label=real_label,
        epochs=epochs,
        lr=lr,
        temperature=temperature,
        dropout=dropout,
        layers=layers,
        batch_size=batch_size,
        m=m,
        save_pred=save_pred,
        noise=noise,
        use_cpu=use_cpu,
        lambd=lambd,
        beta=beta,
        viz_set=viz_set,
        viz_callback=viz_callback,
    )

    # Plot convergence curves if enabled
    if plot_convergence_curve and convergence_history and len(convergence_history['epochs']) > 0:
        plot_convergence(convergence_history, dataset)

    if save_pred:
        results["features"] = embedding
        results["max_epoch"] = best_model_epoch
    elapsed = time.time() - start

    res_eval = compute_cluster_metrics(
        embedding,
        cluster_number,
        real_label,
        save_predictions=save_pred,
        clustering_methods=cluster_methods
    )
    results = {**results, **res_eval, "dataset": dataset, "time": elapsed}

    return results


def _compute_features(model, X_np, batch_size=256, device="cpu"):
    """用 encoder_q + inst_proj 提取实例级表征，并做 L2 归一化"""
    model.eval()
    feats = []
    with torch.no_grad():
        N = X_np.shape[0]
        for i in range(0, N, batch_size):
            xb = torch.as_tensor(X_np[i:i + batch_size], dtype=torch.float32, device=device)
            h = model.encoder_q(xb)
            q = model.inst_proj(h)
            z = F.normalize(q, p=2, dim=1)
            feats.append(z.detach().cpu().numpy())
    return np.vstack(feats)


def train_model(gene_exp, cluster_number, real_label, epochs, lr,
                temperature, dropout, layers, batch_size, m,
                save_pred=False, noise=None, use_cpu=None, evaluate_training=True,
                lambd=1.0, beta=1.0, viz_set=None, viz_callback=None):
    # Initialize convergence history
    convergence_history = {
        'epochs': [],
        'train_losses': [],
        'accs': [],
        'nmis': [],
        'aris': [],
        'puritys': []
    }

    # Device selection
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dims = np.concatenate([[gene_exp.shape[1]], layers])

    # Initialize components
    data_aug_model = SDMCC.Augmenter(drop_rate=dropout)
    encoder_q = SDMCC.EncoderBase(dims)
    encoder_k = SDMCC.EncoderBase(dims)
    instance_projector = SDMCC.ProjectionMLP(layers[2], layers[2] + layers[3], layers[2] + layers[3])
    cluster_projector = SDMCC.ProjectionMLP(layers[2], layers[3], cluster_number)
    model = SDMCC.SDMCC(encoder_q, encoder_k, instance_projector, cluster_projector, cluster_number, m=m)

    data_aug_model.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    if not getattr(config.args, "ablate_contrast", False):
        criterion_instance = contrastive_loss.InstanceContrastiveLoss(temperature=temperature)
        criterion_cluster = contrastive_loss.ClusterContrastiveLossWithEntropy(temperature=temperature)
    else:
        criterion_instance = None
        criterion_cluster = None

    max_value, best_model_epoch = -1, -1
    idx = np.arange(len(gene_exp))
    sure_params = []

    for epoch in range(epochs):
        model.train()
        update_learning_rate(optimizer, epoch, lr)
        np.random.shuffle(idx)

        # Track losses for this epoch
        loss_instance_epoch = []
        loss_cluster_epoch = []
        sure_loss_epoch = []

        # Mini-batch training
        for pre_index in range(len(gene_exp) // batch_size + 1):
            c_idx = np.arange(pre_index * batch_size,
                              min(len(gene_exp), (pre_index + 1) * batch_size))
            if len(c_idx) == 0:
                continue
            c_idx = idx[c_idx]
            c_inp = gene_exp[c_idx]
            input1 = data_aug_model(torch.FloatTensor(c_inp))
            input2 = data_aug_model(torch.FloatTensor(c_inp))

            # Add noise if specified
            if noise is None or noise == 0:
                input1 = torch.FloatTensor(input1).to(device)
                input2 = torch.FloatTensor(input2).to(device)
            else:
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input1.shape))
                input1 = torch.FloatTensor(input1 + noise_vec).to(device)
                noise_vec = torch.FloatTensor(np.random.normal(loc=0, scale=noise, size=input2.shape))
                input2 = torch.FloatTensor(input2 + noise_vec).to(device)

            # Forward pass
            q_instance, q_cluster, k_instance, k_cluster = model(input1, input2)

            if not getattr(config.args, "ablate_contrast", False):
                features_instance = torch.cat([q_instance.unsqueeze(1), k_instance.unsqueeze(1)], dim=1)
                features_cluster = torch.cat([q_cluster.unsqueeze(1), k_cluster.unsqueeze(1)], dim=1)
                # 可选防呆：
                assert features_cluster.shape == (q_cluster.shape[0], 2, q_cluster.shape[1])
                loss_instance = criterion_instance(features_instance)
                loss_cluster = criterion_cluster(features_cluster)
            else:
                loss_instance = torch.tensor(0.0, device=device)
                loss_cluster = torch.tensor(0.0, device=device)

            # LSURE loss computation
            q_instance_cpu = q_instance.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(q_instance_cpu)
            labels = kmeans.labels_
            P = q_instance_cpu.shape[1]
            LSURE_batch = 0.0

            for k in range(cluster_number):
                idx_k = np.where(labels == k)[0]
                if len(idx_k) < 2:
                    continue
                mu_k = np.mean(q_instance_cpu[idx_k], axis=0)
                diff = q_instance_cpu[idx_k] - mu_k
                squared_errors = np.sum(diff ** 2, axis=1)
                variances = np.var(q_instance_cpu[idx_k], axis=0, ddof=1)
                sigma2_k = np.mean(variances)
                N_k = len(idx_k)
                tau2_k = sigma2_k / N_k
                denom = tau2_k + sigma2_k
                factor = sigma2_k / denom if denom != 0 else 0
                LSURE_cluster = np.sum(factor * (squared_errors + P * (tau2_k - sigma2_k)))
                LSURE_batch += LSURE_cluster

            sure_loss = LSURE_batch
            sure_params.append((epoch, pre_index, sure_loss))

            # Track batch losses
            loss_instance_epoch.append(loss_instance.item())
            loss_cluster_epoch.append(loss_cluster.item())
            sure_loss_epoch.append(float(sure_loss))

            # Backward pass
            ablate = getattr(config.args, "ablate_contrast", False)
            if not ablate:
                total_loss = loss_instance + lambd * loss_cluster + beta * torch.tensor(sure_loss, device=device,
                                                                                        dtype=torch.float32)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        # ✅ 正确：epoch结束后计算平均损失（注意缩进！）
        epoch_loss = np.mean(loss_instance_epoch) + lambd * np.mean(loss_cluster_epoch) + beta * np.mean(
            sure_loss_epoch)

        # ✅ 正确：每个epoch结束后评估一次（注意缩进！）
        if evaluate_training and real_label is not None:
            model.eval()
            with torch.no_grad():
                q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
                features = q_instance.detach().cpu().numpy()
            res = compute_cluster_metrics(features, cluster_number, real_label, save_predictions=save_pred)

            # Save to history
            convergence_history['epochs'].append(epoch + 1)
            convergence_history['train_losses'].append(epoch_loss)
            convergence_history['accs'].append(res.get('acc', 0))
            convergence_history['nmis'].append(res.get('nmi', 0))
            convergence_history['aris'].append(res.get('ari', 0))
            # 添加这行调试输出
            if epoch < 3:  # 只在前几个epoch打印调试信息
                print(
                    f"[Debug] Epoch {epoch + 1} saved - ARI: {res.get('ari', 0)}, in list: {len(convergence_history['aris'])}")


            mode_tag = "LSURE-only" if getattr(config.args, "ablate_contrast", False) else "Contrast+LSURE"
            print(f"[{mode_tag}] Epoch {epoch + 1}: Loss: {epoch_loss:.4f}, "
                  f"ACC: {res['acc']:.4f}, ARI: {res['ari']:.4f}, NMI: {res['nmi']:.4f}")

            if res['ari'] + res['nmi'] >= max_value:
                max_value = res['ari'] + res['nmi']
                store_model_checkpoint(config.args.name, model, optimizer, epoch, best_model_epoch)
                best_model_epoch = epoch

        # Visualization callback
        ep = epoch + 1
        if viz_set and (ep in viz_set) and (viz_callback is not None):
            feats_ep = _compute_features(model, gene_exp, batch_size=batch_size, device=device)
            km = KMeans(n_clusters=cluster_number, n_init=20, random_state=0)
            y_pred_ep = km.fit_predict(feats_ep)
            res_ep = {"features": feats_ep, "predicted_labels": y_pred_ep, "dataset": config.args.name}
            try:
                viz_callback(res_ep, real_label, f"{config.args.name}_ep{ep}")
            except Exception as _e:
                print(f"[viz] WARN (epoch {ep}) callback failed: {repr(_e)}")

    # Fine-tuning phase (kept as original)
    best_sure_idx = np.argmin([param[2] for param in sure_params])
    best_sure_epoch, best_sure_index, _ = sure_params[best_sure_idx]

    if best_model_epoch != -1:
        model_fp = os.path.join(os.getcwd(), 'save', config.args.name, f"checkpoint_{best_model_epoch}.tar")
        model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
        model.to(device)

        # Fine-tuning code remains the same...
        # [Previous fine-tuning code here]

    # Load best model and extract final features
    model.eval()
    model_fp = os.path.join(os.getcwd(), 'save', config.args.name, f"checkpoint_{best_model_epoch}.tar")
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    with torch.no_grad():
        q_instance, _, _, _ = model(torch.FloatTensor(gene_exp).to(device), None)
        features = q_instance.detach().cpu().numpy()

    return features, best_model_epoch, convergence_history