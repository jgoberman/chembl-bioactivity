"""Compute Morgan (ECFP4) fingerprints for the processed compound set.

Run after src.preprocessing:

    python -m src.features                            # uses TARGET_CHEMBL_ID
    python -m src.features CHEMBL240

Reads  data/processed/<target>_pic50.csv
Writes data/processed/<target>_morgan2_2048.npz

The .npz file contains:
  fingerprints      — shape (n, 2048), uint8, one row per compound
  canonical_smiles  — shape (n,), the SMILES key (same order as rows)

Morgan / ECFP4 is the standard QSAR baseline: each atom's circular
neighborhood is hashed to one of 2048 bits. Radius=2 means "atom + neighbors
+ neighbors-of-neighbors," which roughly corresponds to ECFP4 in the original
Rogers & Hahn (2010) paper (diameter = 2 * radius).

Why bother caching: a ridge or gradient-boosted model trains in seconds on
9k compounds, but recomputing 9k fingerprints in the hot path is ~10s of
wasted setup per experiment. We compute once, then every model run mmaps
the .npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from src.config import PROCESSED_DIR, TARGET_CHEMBL_ID

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 2
FP_SIZE = 2048


def _generator() -> rdFingerprintGenerator.FingerprintGenerator64:
    return rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)


def compute_fingerprints(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    """Return (fp_matrix, failed_idx).

    fp_matrix has shape (len(smiles), FP_SIZE) with rows of all zeros at any
    index where SMILES could not be parsed (so the matrix stays aligned to
    the input list). failed_idx lists those positions so the caller can drop
    them.
    """
    gen = _generator()
    fps = np.zeros((len(smiles), FP_SIZE), dtype=np.uint8)
    failed: list[int] = []
    for i, smi in enumerate(tqdm(smiles, unit="mol", desc="Morgan FPs")):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed.append(i)
            continue
        fps[i] = gen.GetFingerprintAsNumPy(mol)
    return fps, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", default=TARGET_CHEMBL_ID)
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    in_path = args.processed or PROCESSED_DIR / f"{args.target}_pic50.csv"
    out_path = args.out or PROCESSED_DIR / f"{args.target}_morgan{FP_RADIUS}_{FP_SIZE}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"error: {in_path} not found — run `python -m src.preprocessing` first.",
              file=sys.stderr)
        return 1

    df = pd.read_csv(in_path)
    print(f"Featurizing {args.target}: {len(df):,} compounds")

    fps, failed = compute_fingerprints(df["canonical_smiles"].tolist())
    if failed:
        # SMILES that survived preprocessing should all parse — flag loudly if not.
        print(f"  WARNING: {len(failed)} SMILES failed to parse (indices: "
              f"{failed[:5]}{'...' if len(failed) > 5 else ''}); dropping them.")
        keep = np.ones(len(df), dtype=bool)
        keep[failed] = False
        fps = fps[keep]
        smiles_list = df["canonical_smiles"].to_numpy()[keep].tolist()
    else:
        smiles_list = df["canonical_smiles"].tolist()

    # Fixed-width unicode (not object dtype) so the .npz loads without
    # allow_pickle=True downstream. Size to the longest SMILES + slack.
    max_len = max(len(s) for s in smiles_list)
    smiles = np.array(smiles_list, dtype=f"U{max_len}")

    np.savez_compressed(out_path, fingerprints=fps, canonical_smiles=smiles)

    nbytes = out_path.stat().st_size
    on_bits_per_mol = fps.sum(axis=1).mean()
    print(f"  shape: {fps.shape}  dtype: {fps.dtype}")
    print(f"  mean on-bits per molecule: {on_bits_per_mol:.1f} / {FP_SIZE}  "
          f"(density {on_bits_per_mol / FP_SIZE:.1%})")
    print(f"  wrote {out_path.relative_to(PROCESSED_DIR.parent.parent)} "
          f"({nbytes / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
