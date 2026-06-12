"""Pull bioactivity records for a single ChEMBL target.

Run as a module from the project root:

    python -m src.chembl_client                       # uses TARGET_CHEMBL_ID from config
    python -m src.chembl_client CHEMBL240             # override target

Writes data/raw/<target>_activities.csv with one row per (molecule, assay)
measurement that passed the standard-type / unit filter. Downstream cleanup
(SMILES canonicalization, salt stripping, dedup, pIC50 conversion) lives in
preprocessing.py — this module's only job is a faithful pull.

Uses the ChEMBL REST API directly with retry-on-5xx, since the public service
returns transient 500s often enough that an un-retried pull is unreliable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from src.config import RAW_DIR, STANDARD_TYPES, STANDARD_UNITS, TARGET_CHEMBL_ID

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000  # API max

# Fields we keep from the activities endpoint. Everything else is dropped at
# write time — the raw CSV gets big fast otherwise.
KEEP_COLS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_chembl_id",
    "assay_description",
    "target_chembl_id",
    "document_chembl_id",
]


def _session() -> requests.Session:
    """A requests session that retries 5xx and 429 with exponential backoff."""
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=1.5,            # 0, 1.5, 3, 6, 12, 24 s between retries
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def fetch_activities(target_chembl_id: str) -> pd.DataFrame:
    """Pull all activities for a target, filtered to the configured endpoints."""
    session = _session()
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type__in": ",".join(STANDARD_TYPES),
        "standard_units": STANDARD_UNITS,
        "standard_value__isnull": "false",
        "limit": PAGE_SIZE,
        "offset": 0,
    }

    # First page tells us total_count, which we use to size the progress bar.
    first = session.get(f"{CHEMBL_BASE}/activity.json", params=params, timeout=60)
    first.raise_for_status()
    payload = first.json()
    total = payload["page_meta"]["total_count"]
    rows: list[dict] = list(payload["activities"])

    with tqdm(total=total, initial=len(rows), unit="row", desc="ChEMBL pull") as bar:
        next_url = payload["page_meta"]["next"]
        while next_url:
            resp = session.get(f"https://www.ebi.ac.uk{next_url}", timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            page = payload["activities"]
            rows.extend(page)
            bar.update(len(page))
            next_url = payload["page_meta"]["next"]

    if not rows:
        return pd.DataFrame(columns=KEEP_COLS)
    return pd.DataFrame(rows).reindex(columns=KEEP_COLS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "target",
        nargs="?",
        default=TARGET_CHEMBL_ID,
        help=f"Target ChEMBL ID (default: {TARGET_CHEMBL_ID})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: data/raw/<target>_activities.csv)",
    )
    args = parser.parse_args(argv)

    out_path: Path = args.out or RAW_DIR / f"{args.target}_activities.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching activities for {args.target} ...", flush=True)
    df = fetch_activities(args.target)
    print(f"  pulled {len(df):,} rows")

    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path.relative_to(RAW_DIR.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
