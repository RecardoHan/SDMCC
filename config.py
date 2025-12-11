import argparse

# 帮助文档，快速定参
def get_args():
    parser = argparse.ArgumentParser(
        prog="SDMCC",
        description="SDMCC: Sample-wise Debiased Multilevel Contrastive Clustering Integrating Pattern Mining for Unsupervised Analysis of Single-cell Gene Expression Profiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 文件
    parser.add_argument("--name", type=str, default="10X_PBMC", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="", help="Full path to .h5ad/.mat/.h5 file, overrides --name")
    parser.add_argument("--label_col", type=str, default="", help="Label column in .obs for clustering (e.g., cluster_number)")
    # Cuda
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    # seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    # epoch
    parser.add_argument("--epoch", type=int, default=200, help="Number of training epochs (legacy, see --max_epoch)")
    parser.add_argument("--max_epoch", type=int, default=200, help="Number of training epochs (recommended, overrides --epoch)")
    # 取 top 变异度的 2000 个基因
    parser.add_argument("--select_gene", type=int, default=2000, help="Number of genes to select")
    # mini-batch样本数
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size")
    # Dropout
    parser.add_argument("--dropout", type=float, default=0.9, help="Dropout rate")
    # 学习率
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate")
    # 动量系数
    parser.add_argument("--m", type=float, default=0.5, help="Momentum coefficient")
    # 噪声
    parser.add_argument("--noise", type=float, default=0.1, help="Noise scale")
    # 温度参数
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature parameter")
    # 三层编码
    parser.add_argument("--enc_1", type=int, default=512, help="Dimension of first encoder layer")
    parser.add_argument("--enc_2", type=int, default=256, help="Dimension of second encoder layer")
    parser.add_argument("--enc_3", type=int, default=128, help="Dimension of third encoder layer")
    # 最终输出
    parser.add_argument("--mlp_dim", type=int, default=64, help="Dimension of MLP output")
    # 聚类
    parser.add_argument("--cluster_methods", type=str, default="KMeans", help="Clustering method to use")
    parser.add_argument("--lambd", type=float, default=0.1, help="Weight for cluster contrastive loss (Lclust)")
    parser.add_argument("--beta", type=float, default=0.01, help="Weight for LSURE loss (Llsure)")
    parser.add_argument('--ablate_contrast', action='store_true',
                        help='If set, disable dual contrastive loss for ablation study')

    return parser.parse_args()

args = get_args()
