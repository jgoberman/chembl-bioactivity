# ChEMBL Bioactivity Prediction

End-to-end project: pull bioactivity data from [ChEMBL](https://www.ebi.ac.uk/chembl/) for a single protein target, build a clean compound–activity dataset, and train a model to predict potency (pIC50) from molecular structure.

## Why this dataset

ChEMBL is the canonical open database of bioactive molecules with drug-like properties — ~2.4M compounds, ~20M measured activities, curated from primary medicinal-chemistry literature. It is the standard benchmark source for QSAR (Quantitative Structure–Activity Relationship) work, so results here are directly comparable to published baselines.

The domain framing matters: a "good" model isn't just one with low RMSE on a random split. Compounds in ChEMBL are heavily clustered by chemical series (a single paper often contributes 50+ close analogs), so a random split leaks structure between train and test. The honest evaluation here uses a **scaffold split** — held-out compounds share no Bemis–Murcko scaffold with training compounds — which is much closer to the real-world question "can this model score a new chemotype?"

## Target

Default target: **VEGFR2** (`CHEMBL279`) — a well-studied kinase with ~9k reported IC50 measurements. Configurable in [src/config.py](src/config.py).

## Project structure

```
chembl_bioactivity/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py          # target ChEMBL ID, paths, thresholds
│   ├── chembl_client.py   # data-loading: pull activities for a target
│   ├── preprocessing.py   # SMILES cleanup, unit harmonization, pIC50
│   ├── features.py        # Morgan fingerprints (RDKit)
│   ├── splits.py          # scaffold split (Bemis–Murcko)
│   ├── models.py          # baseline + main model
│   └── evaluation.py      # RMSE, R^2, Spearman, per-scaffold-bin metrics
├── data/
│   ├── raw/               # raw ChEMBL pulls (CSV)
│   └── processed/         # cleaned, deduplicated, fingerprinted
├── notebooks/             # exploration, model diagnostics, story
└── tests/
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Pull bioactivity data for the configured target (CHEMBL279 by default)
python -m src.chembl_client

# 2. Clean + featurize (next step — not yet implemented)
# python -m src.preprocessing

# 3. Train + evaluate with scaffold split
# python -m src.models
```

## Pipeline (planned)

1. **Pull** — `src/chembl_client.py` queries ChEMBL via `chembl_webresource_client`, filters to `IC50` measurements with `standard_units=='nM'` and a defined `standard_value`, writes one row per (molecule, assay) to `data/raw/<target>_activities.csv`.
2. **Clean** — standardize SMILES (RDKit), drop salts/mixtures, deduplicate by canonical SMILES (median pIC50 when multiple measurements exist), convert nM IC50 → pIC50 = 9 − log10(IC50_nM).
3. **Featurize** — Morgan fingerprints (ECFP4, radius=2, 2048 bits) as the baseline; optionally physchem descriptors.
4. **Split** — Bemis–Murcko scaffold split, 80/10/10 train/val/test. Random split reported alongside as a *contrast*, never as the headline number.
5. **Model** — baseline ridge regression on fingerprints; main model gradient-boosted trees. (Stretch: small graph NN.)
6. **Evaluate** — RMSE / R² / Spearman ρ on the scaffold-test set. Report per-scaffold-cluster performance to show where the model fails.

## Honest evaluation

The story this project tells is *not* "I got R²=0.7 on ChEMBL." It is:

- Here is the dataset I built and what I had to throw away (and why).
- Here is what the model learns on a random split (the easy number).
- Here is what it learns on a scaffold split (the honest number, usually much lower).
- Here are the chemical series where the model breaks, and a hypothesis for why.

That contrast is the whole point.

## References

- ChEMBL: https://www.ebi.ac.uk/chembl/
- `chembl_webresource_client` docs: https://github.com/chembl/chembl_webresource_client
- Bemis & Murcko, *J. Med. Chem.* 1996 — scaffold definition
- RDKit: https://www.rdkit.org/
