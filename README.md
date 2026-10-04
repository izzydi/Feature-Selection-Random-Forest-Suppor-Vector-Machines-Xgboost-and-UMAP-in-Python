# Feature Selection, Classification and UMAP in Python

A machine-learning project that combines **feature selection, classical classifiers and dimensionality reduction** on a high-dimensional classification dataset.

## Project overview

The analysis compares multiple modelling tools and uses UMAP to inspect structure in the feature space. The notebook brings together supervised classification, feature ranking/selection and low-dimensional visualization in a single workflow.

## Repository contents

- [`feature_selection_umap_classification.ipynb`](feature_selection_umap_classification.ipynb) — complete Jupyter notebook.
- [`requirements.txt`](requirements.txt) — Python dependencies.
- [`.gitignore`](.gitignore) — local Python/Jupyter exclusions.

## Methods and tools

The workflow uses:

- `pandas` and `NumPy`,
- `scikit-learn`,
- `xgboost`,
- `umap-learn`,
- `matplotlib` and `seaborn`.

Methods include Random Forest, Support Vector Machines, XGBoost, univariate feature selection, scaling and quantile transformation, confusion matrices, classification metrics and UMAP-based dimensionality reduction.

## Data requirements

The project operates on a high-dimensional source dataset that is not included in this repository. Reproduction therefore requires access to the original data and updating any legacy local data path in the notebook.

## Reproducing the analysis

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `feature_selection_umap_classification.ipynb`, update the source-data path if required and run the notebook sequentially.

## Scope

This repository is a portfolio example of exploratory model comparison, feature-selection techniques and manifold visualization in Python.
