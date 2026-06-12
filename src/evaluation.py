"""Report test-set metrics from saved predictions.

Run after src.models:

    python -m src.evaluation                          # uses TARGET_CHEMBL_ID
    python -m src.evaluation CHEMBL240

Reads  data/processed/<target>_predictions.csv
Prints a side-by-side comparison of random-split vs scaffold-split
performance for each model, and a per-scaffold-cluster-size breakdown for
the scaffold-split test set.

The point of the comparison is *not* "model A beats model B." It is:
how much of the apparent test performance is just memorization of close
analogs (random split) vs. true generalization to unseen chemotypes
(scaffold split). A large random→scaffold gap means the model is leaning
heavily on series-level patterns and will probably disappoint on a new
chemotype in the lab.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROCESSED_DIR, TARGET_CHEMBL_ID


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"),
                "r2": float("nan"), "spearman": float("nan")}
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "n": len(y_true),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho),
    }


def _row(label: str, m: dict) -> str:
    return (f"  {label:<28} n={m['n']:>5,}  "
            f"RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  "
            f"R²={m['r2']:>+5.3f}  ρ={m['spearman']:>+5.3f}")


def _headline(preds: pd.DataFrame) -> None:
    print("=" * 72)
    print("Test-set metrics by (split_type, model)")
    print("=" * 72)
    test = preds[preds["split"] == "test"]
    for split_type in ("random", "scaffold"):
        print(f"\n[{split_type} split]")
        for model in ("ridge", "gbm"):
            sub = test[(test.split_type == split_type) & (test.model == model)]
            print(_row(f"{model}", _metrics(sub["y_true"].values, sub["y_pred"].values)))

    print()
    print("Train-vs-test RMSE gap (overfitting check, scaffold split, gbm)")
    sub = preds[(preds.split_type == "scaffold") & (preds.model == "gbm")]
    for split in ("train", "val", "test"):
        s = sub[sub.split == split]
        m = _metrics(s["y_true"].values, s["y_pred"].values)
        print(_row(f"{split}", m))


def _tanimoto_nn(test_fps: np.ndarray, train_fps: np.ndarray) -> np.ndarray:
    """For each test fingerprint, return Tanimoto similarity to its
    nearest neighbor in the training set. Inputs are uint8 0/1 matrices.

    Tanimoto(a, b) = |a ∩ b| / |a ∪ b|. Computed in batches of test rows so
    the intermediate (batch × n_train) matrices stay small.
    """
    train_pop = train_fps.sum(axis=1)               # |b| for each train row
    out = np.empty(len(test_fps), dtype=np.float32)
    batch = 64
    for i in range(0, len(test_fps), batch):
        chunk = test_fps[i : i + batch].astype(np.uint16)
        # inter[j, k] = popcount(chunk[j] AND train_fps[k])
        inter = chunk @ train_fps.T.astype(np.uint16)
        union = chunk.sum(axis=1, keepdims=True) + train_pop - inter
        sims = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        out[i : i + batch] = sims.max(axis=1)
    return out


def _tanimoto_nn_breakdown(preds: pd.DataFrame, target: str) -> None:
    """Bin scaffold-split test compounds by Tanimoto similarity to their
    nearest training neighbor. This is the standard QSAR domain-of-applicability
    lens: a test compound with low max-Tanimoto to training has no close
    analog the model can lean on, and we expect performance to degrade there.

    (We don't break down by scaffold cluster size because the scaffold split,
    by construction, packs all singleton scaffolds into the test set — every
    test compound is a singleton, so cluster-size binning carries no info.)
    """
    print()
    print("=" * 72)
    print("Scaffold-split test set, binned by Tanimoto similarity to nearest")
    print("training compound (gbm model)")
    print("=" * 72)

    fps_npz = np.load(PROCESSED_DIR / f"{target}_morgan2_2048.npz", allow_pickle=False)
    fp_smiles = pd.Series(np.arange(len(fps_npz["canonical_smiles"])),
                          index=fps_npz["canonical_smiles"])
    fps = fps_npz["fingerprints"]

    sub = preds[(preds.split_type == "scaffold") & (preds.model == "gbm")].copy()
    train_idx = fp_smiles.reindex(sub.loc[sub.split == "train", "canonical_smiles"]).to_numpy()
    test_rows = sub[sub.split == "test"].reset_index(drop=True)
    test_idx = fp_smiles.reindex(test_rows["canonical_smiles"]).to_numpy()

    nn_sim = _tanimoto_nn(fps[test_idx], fps[train_idx])
    test_rows["nn_tanimoto"] = nn_sim

    print(f"  test-set NN Tanimoto:  min={nn_sim.min():.2f}  "
          f"median={np.median(nn_sim):.2f}  max={nn_sim.max():.2f}")
    print()
    bins = [(0.00, 0.30, "very dissimilar  (≤0.30)"),
            (0.30, 0.50, "dissimilar       (0.30–0.50)"),
            (0.50, 0.70, "moderate         (0.50–0.70)"),
            (0.70, 1.01, "similar          (>0.70)")]
    for lo, hi, name in bins:
        b = test_rows[(test_rows.nn_tanimoto >= lo) & (test_rows.nn_tanimoto < hi)]
        print(_row(name, _metrics(b["y_true"].values, b["y_pred"].values)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", default=TARGET_CHEMBL_ID)
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args(argv)

    pred_path = args.predictions or PROCESSED_DIR / f"{args.target}_predictions.csv"
    if not pred_path.exists():
        print(f"error: {pred_path} not found — run `python -m src.models` first.",
              file=sys.stderr)
        return 1

    preds = pd.read_csv(pred_path)
    _headline(preds)
    _tanimoto_nn_breakdown(preds, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
