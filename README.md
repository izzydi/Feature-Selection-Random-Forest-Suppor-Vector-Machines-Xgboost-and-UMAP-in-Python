# Feature Selection, Classification and UMAP in Python

A high-dimensional machine-learning project combining **training-only preprocessing, feature selection, tree-based classification and supervised UMAP**.

## Primary workflow

[`src/feature_selection_pipeline.py`](src/feature_selection_pipeline.py) is the audited implementation. It uses a stratified train/test split, fits robust scaling, quantile transformation and `SelectKBest` on training data only, compares Random Forest and XGBoost on the held-out test set, and evaluates XGBoost on a supervised UMAP representation learned from training data only.

Optional validation datasets are transformed with the same fitted preprocessing, selector and UMAP objects; they are never used to fit those transformations.

## Repository structure

```text
.
├── src/
│   └── feature_selection_pipeline.py
├── data/
│   └── README.md
├── archive/
│   ├── README.md
│   └── legacy_feature_selection_exploration.ipynb
├── requirements.txt
└── README.md
```

## Methods and tools

- `pandas` and `NumPy` for data handling
- `scikit-learn` for splitting, preprocessing, feature selection and Random Forest
- `xgboost` for gradient-boosted classification
- `umap-learn` for supervised dimensionality reduction

The audited workflow reports accuracy, balanced accuracy and weighted F1 on the untouched hold-out set and writes a machine-readable summary to `outputs/metrics.json`.

## Legacy notebook

The original notebook is retained under [`archive/`](archive/) for transparency. It contains historical machine-specific paths and leakage-prone preprocessing choices, so it is no longer presented as the recommended implementation.

## Data

Raw data are not committed. See [`data/README.md`](data/README.md) for the expected 112-predictor-plus-target schema.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/feature_selection_pipeline.py
```

## Scope

This repository demonstrates reproducible feature selection, model comparison and out-of-sample manifold transformation. It is a portfolio project rather than a production scoring service.
