# Feature Selection, Classification and UMAP in Python

A machine-learning notebook that combines **feature selection, classical classifiers and dimensionality reduction** on a high-dimensional classification dataset.

## Project overview

The analysis compares multiple modelling tools and uses UMAP to inspect structure in the feature space. The notebook brings together supervised classification, feature ranking/selection and low-dimensional visualization in a single workflow.

## Repository contents

- [`v_02.ipynb`](v_02.ipynb) — complete Jupyter notebook.

## Methods and tools

The notebook uses Python libraries including:

- `pandas` and `NumPy`,
- `scikit-learn`,
- `xgboost`,
- `umap-learn`,
- `matplotlib` and `seaborn`.

The workflow includes tools such as:

- Random Forest,
- Support Vector Machines,
- XGBoost,
- univariate feature selection,
- scaling and quantile transformation,
- confusion matrices and classification metrics,
- UMAP-based dimensionality reduction and visualization.

## Data requirements

The notebook operates on a high-dimensional dataset that is not included in this repository. Reproduction requires access to the original source data and may require updating local file paths in the notebook.

## Reproducing the analysis

1. Install Python and Jupyter.
2. Install the required packages (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `umap-learn`, `matplotlib`, `seaborn`).
3. Update the dataset path where necessary.
4. Run `v_02.ipynb` from top to bottom.

## Scope

This repository is a portfolio example of exploratory model comparison, feature-selection techniques and manifold visualization in Python.
