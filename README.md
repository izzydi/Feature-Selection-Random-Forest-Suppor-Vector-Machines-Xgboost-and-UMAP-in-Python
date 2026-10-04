# Feature Selection, Classification and UMAP in Python

A high-dimensional machine-learning project combining **training-only preprocessing, feature selection, tree-based classification and supervised UMAP**.

## Primary workflow

[`src/feature_selection_pipeline.py`](src/feature_selection_pipeline.py) is the audited implementation. It uses a stratified train/test split, fits robust scaling, quantile transformation and `SelectKBest` on training data only, compares fixed-configuration Random Forest and XGBoost baselines on the held-out test set, and evaluates a fixed-configuration XGBoost model on a supervised UMAP representation learned from training data only.

Optional validation datasets are transformed with the same fitted preprocessing, selector and UMAP objects; they are never used to fit those transformations.

## Repository structure

```text
.
├── .github/workflows/ci.yml          # Python 3.12 CI
├── src/
│   └── feature_selection_pipeline.py
├── tests/
│   └── test_smoke.py                 # schema/metric smoke tests
├── data/
│   └── README.md                     # schema and provenance note
├── archive/
│   ├── README.md
│   └── legacy_feature_selection_exploration.ipynb
├── requirements.txt                 # pinned dependencies
└── README.md
```

## Methods and evaluation

- `pandas` and `NumPy` for data handling
- `scikit-learn` for splitting, preprocessing, feature selection and Random Forest
- `xgboost` for gradient-boosted classification
- `umap-learn` for supervised dimensionality reduction

The audited workflow reports accuracy, balanced accuracy and weighted F1 on the untouched hold-out set and writes a machine-readable summary to `outputs/metrics.json`.

The current Random Forest and XGBoost hyperparameters are **deliberately fixed baseline configurations**. They are not presented as cross-validated or globally optimized settings; the project focuses on leakage-safe feature/representation comparisons rather than claiming optimized benchmark performance.

## Reproducibility and CI

Direct Python dependencies are pinned in [`requirements.txt`](requirements.txt). GitHub Actions creates a clean Python 3.12 environment, installs the pinned environment, compiles the source and runs synthetic smoke tests on every push and pull request. The tests do not require the unavailable raw wave-function data.

## Legacy notebook

The original notebook is retained under [`archive/`](archive/) for transparency. It contains historical machine-specific paths and leakage-prone preprocessing choices, so it is no longer presented as the recommended implementation.

## Data

Raw data are not committed and the historical materials do not provide a stable public source/version/checksum. See [`data/README.md`](data/README.md) for the expected 112-predictor-plus-target schema and provenance limitation.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/feature_selection_pipeline.py
```

## Scope

This repository demonstrates reproducible feature selection, baseline model comparison and out-of-sample manifold transformation. It is a portfolio project rather than a production scoring service.
