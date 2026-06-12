"""Build a scaffold split and a random split for the processed compound set.

Run after src.preprocessing:

    python -m src.splits                              # uses TARGET_CHEMBL_ID
    python -m src.splits CHEMBL240

Reads  data/processed/<target>_pic50.csv
Writes data/processed/<target>_splits.csv

Output columns: canonical_smiles, scaffold, split_scaffold, split_random
Downstream code joins this file onto the labels file on canonical_smiles.

Scaffold split — what and why:
  Compounds in ChEMBL cluster heavily by chemical series: a single paper often
  contributes 50+ close analogs of the same core. A random split leaks that
  core between train and test, so the model is graded on its ability to fit
  a chemotype it has already memorized — not its ability to score a *new*
  chemotype. The Bemis-Murcko scaffold (Bemis & Murcko, J. Med. Chem. 1996)
  reduces a molecule to its ring system + linkers; grouping compounds by
  scaffold and assigning whole groups to a single split puts genuinely
  unseen chemistry in val and test.

  We sort scaffold groups largest-first and greedily fill train, then val,
  then test (the standard DeepChem-style heuristic). Train gets the big
  popular series, val and test get rarer / singleton scaffolds — which is
  the harder, more honest evaluation.

The random split is generated alongside as a *contrast*, not a baseline to
beat. The headline metric is the scaffold-split number; the random-split
number is shown to quantify how much of the apparent performance is just
memorization of close analogs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.config import PROCESSED_DIR, TARGET_CHEMBL_ID

RDLogger.DisableLog("rdApp.*")

FRAC_TRAIN = 0.80
FRAC_VAL = 0.10
FRAC_TEST = 0.10
RANDOM_SEED = 17


def murcko_scaffold(smiles: str) -> str:
    """Bemis–Murcko scaffold as canonical SMILES. Empty string for acyclic mols."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(scaffolds: pd.Series) -> np.ndarray:
    """Return an array of split labels ('train' | 'val' | 'test') aligned to scaffolds.

    Whole scaffold groups go to a single split. Groups are sorted largest-first
    and packed greedily: fill train to capacity, then val, then test.
    """
    n = len(scaffolds)
    n_train_cap = int(n * FRAC_TRAIN)
    n_val_cap = int(n * FRAC_VAL)

    # scaffold -> positional indices into the original Series
    groups: dict[str, list[int]] = {}
    for i, scaf in enumerate(scaffolds):
        groups.setdefault(scaf, []).append(i)
    sorted_groups = sorted(groups.values(), key=len, reverse=True)

    labels = np.empty(n, dtype=object)
    n_train = n_val = 0
    for grp in sorted_groups:
        if n_train + len(grp) <= n_train_cap:
            labels[grp] = "train"
            n_train += len(grp)
        elif n_val + len(grp) <= n_val_cap:
            labels[grp] = "val"
            n_val += len(grp)
        else:
            labels[grp] = "test"
    return labels


def random_split(n: int, seed: int) -> np.ndarray:
    """80/10/10 random split, reproducible with seed."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * FRAC_TRAIN)
    n_val = int(n * FRAC_VAL)

    labels = np.empty(n, dtype=object)
    labels[idx[:n_train]] = "train"
    labels[idx[n_train : n_train + n_val]] = "val"
    labels[idx[n_train + n_val :]] = "test"
    return labels


def _summarize(df: pd.DataFrame) -> None:
    n = len(df)
    n_scaf = df["scaffold"].nunique()
    n_singleton = (df.groupby("scaffold").size() == 1).sum()
    largest = df.groupby("scaffold").size().max()
    print(f"  unique scaffolds:        {n_scaf:>5,}  "
          f"(singletons: {n_singleton:,}; largest cluster: {largest:,} compounds)")

    for kind in ("scaffold", "random"):
        col = f"split_{kind}"
        counts = df[col].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int)
        means = df.groupby(col)["pIC50"].mean().reindex(["train", "val", "test"])
        print(f"  {kind:>8} split:  "
              f"train={counts['train']:>5,}  val={counts['val']:>4,}  test={counts['test']:>4,}  "
              f"| pIC50 means: {means['train']:.2f} / {means['val']:.2f} / {means['test']:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", default=TARGET_CHEMBL_ID)
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    in_path = args.processed or PROCESSED_DIR / f"{args.target}_pic50.csv"
    out_path = args.out or PROCESSED_DIR / f"{args.target}_splits.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"error: {in_path} not found — run `python -m src.preprocessing` first.",
              file=sys.stderr)
        return 1

    df = pd.read_csv(in_path)
    print(f"Splitting {args.target}: {len(df):,} compounds")

    df["scaffold"] = df["canonical_smiles"].map(murcko_scaffold)
    df["split_scaffold"] = scaffold_split(df["scaffold"])
    df["split_random"] = random_split(len(df), RANDOM_SEED)

    _summarize(df)

    out = df[["canonical_smiles", "scaffold", "split_scaffold", "split_random"]]
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path.relative_to(PROCESSED_DIR.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
