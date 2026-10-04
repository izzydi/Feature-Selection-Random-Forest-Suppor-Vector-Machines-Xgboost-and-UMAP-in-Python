# Legacy exploration

`legacy_feature_selection_exploration.ipynb` is preserved as a historical exploratory notebook.

It is not the recommended evaluation workflow. The legacy notebook contains machine-specific data paths and several preprocessing cells that fit transformations separately on held-out or validation data. Those choices can leak information and make performance estimates difficult to interpret.

Use [`../src/feature_selection_pipeline.py`](../src/feature_selection_pipeline.py) for the audited implementation. It fits preprocessing and feature selection on training data only, uses a stratified hold-out set, and reuses the same fitted transformations for external validation.
