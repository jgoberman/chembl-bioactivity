"""Train baseline + main models on both splits; save predictions for evaluation.

Run after src.features:

    python -m src.models                              # uses TARGET_CHEMBL_ID
    python -m src.models CHEMBL240

Reads:
  data/processed/<target>_pic50.csv
  data/processed/<target>_splits.csv
  data/processed/<target>_morgan2_2048.npz

Writes:
  data/processed/<target>_predictions.csv
    columns: canonical_smiles, scaffold, y_true, model, split_type, y_pred
    One row per (compound, model, split_type). Includes train/val/test —
    the 'split' column from the splits file is preserved so evaluation can
    filter to test.

Two models:
  * ridge — RidgeCV over a fixed alpha grid. Linear-on-fingerprint baseline.
  * gbm   — HistGradientBoostingRegressor. Non-linear main model. Early
            stopping uses an internal validation slice of the train set
            (sklearn handles this), so we still get an honest test number.

Two splits:
  * random   — the easy contrast (analog memorization permitted)
  * scaffold — the honest headline number (held-out chemotypes)

Models are intentionally not tuned per split: same hyperparameters across
both so the only thing that differs is the data partition. The gap between
random-split and scaffold-split test error is then attributable to the split
strategy, not to model selection bias.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

from src.config import PROCESSED_DIR, TARGET_CHEMBL_ID

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
GBM_PARAMS = dict(
    max_iter=500,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=17,
)


def _load(target: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Join labels + splits + fingerprints into a single aligned table.

    Returns (df, fp_matrix) where df has columns
        canonical_smiles, pIC50, scaffold, split_scaffold, split_random
    and fp_matrix is shape (len(df), 2048), row-aligned to df.
    """
    labels = pd.read_csv(PROCESSED_DIR / f"{target}_pic50.csv")
    splits = pd.read_csv(PROCESSED_DIR / f"{target}_splits.csv")
    fps_npz = np.load(PROCESSED_DIR / f"{target}_morgan2_2048.npz", allow_pickle=False)

    fp_smiles = pd.Series(fps_npz["canonical_smiles"])
    fp_matrix = fps_npz["fingerprints"]

    # Join labels + splits on canonical_smiles.
    df = labels.merge(splits, on="canonical_smiles", how="inner")
    # Reindex fingerprints into df's order. SMILES is unique by construction
    # (we deduped on it in preprocessing), so this is well-defined.
    fp_pos = pd.Series(np.arange(len(fp_smiles)), index=fp_smiles).reindex(
        df["canonical_smiles"]
    )
    if fp_pos.isna().any():
        missing = int(fp_pos.isna().sum())
        raise RuntimeError(f"{missing} compounds in labels+splits missing from fingerprints")
    return df, fp_matrix[fp_pos.to_numpy().astype(int)]


def _train_one(name: str, X_train, y_train, X_all):
    """Fit a model and return predictions for every row in X_all."""
    t0 = time.time()
    if name == "ridge":
        model = RidgeCV(alphas=RIDGE_ALPHAS)
    elif name == "gbm":
        model = HistGradientBoostingRegressor(**GBM_PARAMS)
    else:
        raise ValueError(name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_all)
    print(f"    {name:>6}  trained in {time.time() - t0:.1f}s "
          f"(train n={len(y_train):,})")
    return y_pred


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", default=TARGET_CHEMBL_ID)
    args = parser.parse_args(argv)

    df, X = _load(args.target)
    y = df["pIC50"].to_numpy()
    print(f"Loaded {args.target}: {len(df):,} compounds, X shape {X.shape}")

    records: list[pd.DataFrame] = []
    for split_kind in ("random", "scaffold"):
        col = f"split_{split_kind}"
        is_train = df[col].to_numpy() == "train"
        print(f"  split_kind={split_kind}: train n={is_train.sum():,}")
        X_train, y_train = X[is_train], y[is_train]
        for model_name in ("ridge", "gbm"):
            y_pred = _train_one(model_name, X_train, y_train, X)
            records.append(pd.DataFrame({
                "canonical_smiles": df["canonical_smiles"].to_numpy(),
                "scaffold": df["scaffold"].to_numpy(),
                "split": df[col].to_numpy(),
                "split_type": split_kind,
                "model": model_name,
                "y_true": y,
                "y_pred": y_pred,
            }))

    out = pd.concat(records, ignore_index=True)
    out_path = PROCESSED_DIR / f"{args.target}_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path.relative_to(PROCESSED_DIR.parent.parent)} "
          f"({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
