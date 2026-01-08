# SDMCC
SDMCC: Sample-wise Debiased Multilevel Contrastive Clustering Integrating Pattern Mining for Unsupervised Analysis of Single-cell Gene Expression Profiles
---

## 1. Data Preparation

Supported input formats:
- .h5ad (AnnData, recommended)
- .h5
### Data loading priority

1. If `--data_path` is provided, load from the specified path  
2. Otherwise, load from `data/{name}.h5ad`

### Notes
- For `.h5ad`, a label column in `adata.obs` is required for evaluation  
- Use `--label_col` to specify the label column name  

---
## 2. Run
### Basic usage (load from `data/` directory)
```bash
python main.py \
  --name PBMC \
  --max_epoch 100

## 3. Dependencies

This project only relies on common scientific Python packages.

### Required

- Python >= 3.9
- torch
- numpy
- scipy
- scikit-learn
- scanpy
- anndata
- h5py
- pandas
- matplotlib
- umap-learn


## 4.Main Arguments
--name            Dataset name (used for output naming)
--data_path       Full path to data file (overrides --name)
--label_col       Label column name in .h5ad (for evaluation)

--max_epoch       Number of training epochs
--batch_size      Batch size
--lr              Learning rate
--dropout         Feature dropout for augmentation
--temperature     Contrastive learning temperature
--m               Momentum coefficient for key encoder

--lambd           Weight of cluster contrastive loss
--beta            Weight of LSURE loss
--ablate_contrast Disable contrastive loss (ablation study)


