"""Project-wide configuration. Keep simple — one source of truth for paths and target."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Default target: VEGFR2 (Kinase domain). ~9k IC50 measurements in ChEMBL.
# Other reasonable starter targets:
#   CHEMBL240  hERG          (safety / cardiotox)
#   CHEMBL217  Dopamine D2
#   CHEMBL220  Acetylcholinesterase
#   CHEMBL3717 JAK2
TARGET_CHEMBL_ID = "CHEMBL279"

# Activity filter: only keep these endpoints, in nM, with a defined value.
STANDARD_TYPES = ("IC50",)
STANDARD_UNITS = "nM"

# pIC50 = 9 - log10(IC50 in nM).  Drop measurements outside a sane range
# (the literature has plenty of >100 mM "inactive" rows that are noise).
MIN_NM = 0.01      # 10 pM  -> pIC50 = 11
MAX_NM = 1e7       # 10 mM  -> pIC50 = 2
