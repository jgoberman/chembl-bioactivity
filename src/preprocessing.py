"""Clean raw ChEMBL activities into a model-ready (canonical_smiles, pIC50) table.

Run after src.chembl_client:

    python -m src.preprocessing                       # uses TARGET_CHEMBL_ID
    python -m src.preprocessing CHEMBL240             # override target

Reads  data/raw/<target>_activities.csv
Writes data/processed/<target>_pic50.csv  (one row per unique compound)

Pipeline:
  1. Drop rows missing SMILES or standard_value.
  2. Keep only relation == '=' (drop censored '>' / '<' measurements).
  3. Keep IC50 within [MIN_NM, MAX_NM] (drop obvious noise).
  4. Standardize SMILES: parse with RDKit, keep the largest organic fragment
     (strips salts / counter-ions), emit canonical SMILES. Unparseable rows
     are dropped — count is reported.
  5. Compute pIC50 = 9 - log10(IC50_nM).
  6. Deduplicate on canonical SMILES. For each compound we take the median
     pIC50 across replicates and record the spread; rows where the spread
     exceeds DEDUP_SPREAD_LIMIT log units are flagged 'noisy' and dropped,
     since QSAR models are not well-served by self-contradictory labels.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from src.config import MAX_NM, MIN_NM, PROCESSED_DIR, RAW_DIR, TARGET_CHEMBL_ID

# RDKit prints a warning to stderr for every unparseable SMILES. We count
# failures ourselves and report a summary at the end; silence the noise.
RDLogger.DisableLog("rdApp.*")

# A compound's replicate pIC50 measurements wider than this many log units
# get dropped as noisy. 1.0 log = 10x difference in IC50, which is the
# rough reproducibility floor for IC50 assays across labs.
DEDUP_SPREAD_LIMIT = 1.0


def standardize_smiles(smiles: str) -> str | None:
    """Parse, keep largest fragment, canonicalize. None if unparseable."""
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Salts / mixtures arrive as dot-separated fragments. Keep the heaviest one
    # by atom count — the drug-like piece almost always wins; counter-ions like
    # Na+, Cl-, citrate are tiny by comparison.
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())

    # Drop trivially small fragments (likely we kept only a counter-ion because
    # the original SMILES was inorganic).
    if mol.GetNumHeavyAtoms() < 3:
        return None

    try:
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException):
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def ic50_nm_to_pic50(nm: float) -> float:
    return 9.0 - math.log10(nm)


def preprocess(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run the full pipeline. Returns (processed_df, stats_dict)."""
    stats: dict[str, int] = {"raw_rows": len(raw_df)}

    df = raw_df.copy()
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    stats["after_dropna"] = len(df)

    df = df[df["standard_relation"] == "="]
    stats["after_relation_filter"] = len(df)

    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[(df["standard_value"] >= MIN_NM) & (df["standard_value"] <= MAX_NM)]
    stats["after_range_filter"] = len(df)

    df["std_smiles"] = df["canonical_smiles"].map(standardize_smiles)
    df = df.dropna(subset=["std_smiles"])
    stats["after_smiles_standardize"] = len(df)

    df["pIC50"] = df["standard_value"].map(ic50_nm_to_pic50)

    # Dedup: group by canonical SMILES, take median pIC50, keep n_meas and
    # spread (max - min) so we can drop noisy compounds and let downstream
    # see how many replicates each label is averaged from.
    grouped = df.groupby("std_smiles")["pIC50"].agg(
        pIC50="median",
        n_meas="count",
        spread=lambda s: float(s.max() - s.min()) if len(s) > 1 else 0.0,
    )
    stats["unique_compounds"] = len(grouped)

    clean = grouped[grouped["spread"] <= DEDUP_SPREAD_LIMIT].reset_index()
    stats["after_spread_filter"] = len(clean)
    stats["dropped_noisy"] = stats["unique_compounds"] - stats["after_spread_filter"]

    clean = clean.rename(columns={"std_smiles": "canonical_smiles"})
    clean = clean[["canonical_smiles", "pIC50", "n_meas", "spread"]]
    return clean, stats


def _print_stats(stats: dict, target: str) -> None:
    print(f"Preprocessing {target}:")
    print(f"  raw rows:                {stats['raw_rows']:>7,}")
    print(f"  after dropna:            {stats['after_dropna']:>7,}")
    print(f"  after relation == '=':   {stats['after_relation_filter']:>7,}")
    print(f"  after IC50 range filter: {stats['after_range_filter']:>7,}")
    print(f"  after SMILES standardize:{stats['after_smiles_standardize']:>7,}")
    print(f"  unique compounds:        {stats['unique_compounds']:>7,}")
    print(f"  dropped noisy (spread>{DEDUP_SPREAD_LIMIT}): {stats['dropped_noisy']:>7,}")
    print(f"  final compounds:         {stats['after_spread_filter']:>7,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", default=TARGET_CHEMBL_ID)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    raw_path = args.raw or RAW_DIR / f"{args.target}_activities.csv"
    out_path = args.out or PROCESSED_DIR / f"{args.target}_pic50.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"error: {raw_path} not found — run `python -m src.chembl_client` first.",
              file=sys.stderr)
        return 1

    raw_df = pd.read_csv(raw_path)
    clean, stats = preprocess(raw_df)
    _print_stats(stats, args.target)

    clean.to_csv(out_path, index=False)
    print(f"  wrote {out_path.relative_to(PROCESSED_DIR.parent.parent)}")
    print()
    print(f"pIC50 distribution:  min={clean['pIC50'].min():.2f}  "
          f"median={clean['pIC50'].median():.2f}  max={clean['pIC50'].max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
