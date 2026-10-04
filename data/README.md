# Data layout

Place the high-dimensional source dataset here as:

```text
data/
├── wave_functions.csv
└── validation/          # optional
    ├── shock_1.csv
    ├── shock_2.csv
    └── ...
```

The audited pipeline expects at least **113 columns**: the first 112 are predictors and column 113 (zero-based index 112) is the classification target. Extra source columns are ignored deliberately so the feature schema stays fixed.

Raw datasets are not committed because of their size and/or redistribution constraints.
