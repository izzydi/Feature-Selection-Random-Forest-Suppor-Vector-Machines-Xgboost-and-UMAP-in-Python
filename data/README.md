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

The historical project materials do not contain a stable public source URL, version identifier or checksum for the original wave-function files. Raw datasets are therefore not committed because of their size and/or redistribution constraints, and this repository does not claim that the historical raw data can be reconstructed from an arbitrary public download.

If you have the original project files, record their provenance and checksums before reporting reproduced metrics. The code itself uses project-relative paths and applies preprocessing/feature selection learned from training data to held-out and validation datasets unchanged.
