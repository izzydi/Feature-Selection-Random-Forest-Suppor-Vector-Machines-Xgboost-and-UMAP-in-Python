"""Audited classification workflow for high-dimensional wave-function data.

All learned preprocessing and feature selection are fitted on the training set only.
The held-out test set and optional validation datasets are transformed with the
already-fitted objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap.umap_ as umap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, RobustScaler

RANDOM_STATE = 1821
TARGET_COLUMN = 112
EXPECTED_COLUMNS = 113
DEFAULT_SAMPLE_SIZE = 6000

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "wave_functions.csv"
OUTPUT_DIR = ROOT / "outputs"


def load_source(path: Path, *, skiprows: int = 1_700_000, nrows: int = 800_000) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. See data/README.md for the expected layout.")
    df = pd.read_csv(path, header=None, skiprows=skiprows, nrows=nrows)
    if df.shape[1] < EXPECTED_COLUMNS:
        raise ValueError(f"Expected at least {EXPECTED_COLUMNS} columns, found {df.shape[1]}.")
    return df.iloc[:, :EXPECTED_COLUMNS].copy()


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df.iloc[:, :TARGET_COLUMN], df.iloc[:, TARGET_COLUMN].to_numpy()


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def transform_validation(
    path: Path,
    preprocessor,
    selector: SelectKBest,
    label_encoder: LabelEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < EXPECTED_COLUMNS:
        raise ValueError(f"{path} must contain at least {EXPECTED_COLUMNS} columns.")
    df = df.iloc[:, :EXPECTED_COLUMNS]
    x, y_raw = split_xy(df)
    y = label_encoder.transform(y_raw)
    x_scaled = preprocessor.transform(x)
    return selector.transform(x_scaled), y


def main(data_path: Path, sample_size: int) -> None:
    df = load_source(data_path)
    n = min(sample_size, len(df))
    if n < 20:
        raise ValueError("Not enough rows for a meaningful stratified hold-out evaluation.")

    sampled = df.sample(n=n, replace=False, random_state=RANDOM_STATE)
    x, y_raw = split_xy(sampled)

    x_train, x_test, y_train_raw, y_test_raw = train_test_split(
        x,
        y_raw,
        test_size=0.30,
        stratify=y_raw,
        random_state=RANDOM_STATE,
    )

    label_encoder = LabelEncoder().fit(y_train_raw)
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    n_quantiles = min(1000, len(x_train))
    preprocessor = make_pipeline(
        RobustScaler(),
        QuantileTransformer(n_quantiles=n_quantiles, random_state=RANDOM_STATE),
    )
    x_train_scaled = preprocessor.fit_transform(x_train)
    x_test_scaled = preprocessor.transform(x_test)

    selector = SelectKBest(score_func=f_classif, k=min(17, x_train_scaled.shape[1]))
    x_train_selected = selector.fit_transform(x_train_scaled, y_train)
    x_test_selected = selector.transform(x_test_scaled)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=700,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic" if len(label_encoder.classes_) == 2 else "multi:softprob",
            eval_metric="logloss" if len(label_encoder.classes_) == 2 else "mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(x_train_selected, y_train)
        results[name] = score(y_test, model.predict(x_test_selected))

    # Supervised UMAP is fitted on selected training features only.
    manifold = umap.UMAP(
        n_neighbors=min(100, max(2, len(x_train_selected) - 1)),
        min_dist=0.1,
        n_components=2,
        metric="manhattan",
        random_state=RANDOM_STATE,
    ).fit(x_train_selected, y_train)

    train_umap = manifold.transform(x_train_selected)
    test_umap = manifold.transform(x_test_selected)

    umap_xgb = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=1.0,
        objective="binary:logistic" if len(label_encoder.classes_) == 2 else "multi:softprob",
        eval_metric="logloss" if len(label_encoder.classes_) == 2 else "mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    umap_xgb.fit(train_umap, y_train)
    results["xgboost_on_supervised_umap"] = score(y_test, umap_xgb.predict(test_umap))

    validation_dir = ROOT / "data" / "validation"
    validation_results: dict[str, dict[str, float]] = {}
    if validation_dir.exists():
        for path in sorted(validation_dir.glob("*.csv")):
            x_val_selected, y_val = transform_validation(
                path, preprocessor, selector, label_encoder
            )
            x_val_umap = manifold.transform(x_val_selected)
            validation_results[path.name] = score(y_val, umap_xgb.predict(x_val_umap))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "classes": [str(x) for x in label_encoder.classes_],
        "sample_size": n,
        "selected_features": selector.get_support(indices=True).tolist(),
        "held_out_test": results,
        "external_validation": validation_results,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    args = parser.parse_args()
    main(args.data, args.sample_size)
