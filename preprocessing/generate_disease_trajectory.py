"""
Build per-patient disease trajectories capturing age at diagnosis for all
diseases listed in preprocessing/variables.xlsx (disease report time sheet).
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

# Add project root to Python path for cross-directory imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import pandas as pd

from benchmarking.preprocess_survival import (
    DATA_PATH,
    MONTH_FALLBACK,
    PROJECT_ROOT,
    find_columns_for_code,
    load_header,
)

VARIABLES_PATH = PROJECT_ROOT / "preprocessing" / "variables.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "preprocessed" / "disease_trajectory.csv"
DISEASE_SHEET = "disease report time"
BIRTH_YEAR_COL = "p34"
BIRTH_MONTH_COL = "p52"
MIN_VALID_YEAR = 1900


def slugify(name: str) -> str:
    text = (name or "disease").lower()
    text = re.sub(r"[^0-9a-z]+", "_", text).strip("_")
    return text or "disease"


def load_disease_metadata(path: Path) -> List[Dict[str, str]]:
    sheet = pd.read_excel(path, sheet_name=DISEASE_SHEET)
    sheet = sheet[sheet["Field ID"].notna()].copy()
    sheet["field_id"] = sheet["Field ID"].astype(int).astype(str)
    metadata: List[Dict[str, str]] = []
    for _, row in sheet.iterrows():
        field_id = row["field_id"]
        field_name = str(row.get("Field name", "")).strip()
        slug = slugify(field_name)
        column_name = f"{slug}_{field_id}_age"
        metadata.append(
            {
                "field_id": field_id,
                "field_name": field_name or field_id,
                "feature_name": column_name,
            }
        )
    return metadata


def collect_columns_for_fields(header: Sequence[str], metadata: List[Dict[str, str]]) -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = []
    for item in metadata:
        columns = find_columns_for_code(header, item["field_id"])
        if not columns:
            continue
        item = item.copy()
        item["columns"] = columns
        prepared.append(item)
    return prepared


def compute_birth_dates(df: pd.DataFrame) -> pd.Series:
    years = pd.to_numeric(df[BIRTH_YEAR_COL], errors="coerce")
    years = years.where(years >= MIN_VALID_YEAR)
    months = pd.to_numeric(df[BIRTH_MONTH_COL], errors="coerce").round().clip(1, 12)
    months = months.fillna(MONTH_FALLBACK)
    return pd.to_datetime({"year": years, "month": months, "day": 15}, errors="coerce")


def build_age_matrix(chunk: pd.DataFrame, metadata: List[Dict[str, str]]) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["eid"] = chunk["eid"].astype(int)
    birth_dates = compute_birth_dates(chunk)
    rows: Dict[str, pd.Series] = {"eid": chunk["eid"]}

    for item in metadata:
        cols = [col for col in item.get("columns", []) if col in chunk.columns]
        if not cols:
            rows[item["feature_name"]] = pd.Series(pd.NA, index=chunk.index, dtype="float64")
            continue

        date_frame = pd.DataFrame(
            {col: pd.to_datetime(chunk[col], errors="coerce") for col in cols},
            index=chunk.index,
        )
        earliest = date_frame.min(axis=1)
        age_years = (earliest - birth_dates).dt.days / 365.25
        age_years = age_years.where(earliest.notna())
        # Remove non-sensical ages (negative or >120)
        age_years = age_years.where(age_years >= 0)
        age_years = age_years.where(age_years <= 120)
        rows[item["feature_name"]] = age_years

    return pd.DataFrame(rows)


def generate_disease_trajectory(
    data_path: Path,
    variables_path: Path,
    output_path: Path,
    chunk_size: int = 5000,
) -> None:
    metadata = load_disease_metadata(variables_path)
    header = load_header(data_path)
    metadata = collect_columns_for_fields(header, metadata)
    if not metadata:
        raise RuntimeError("No disease fields from variables.xlsx were located in the cohort CSV.")

    required_cols = {"eid", BIRTH_YEAR_COL, BIRTH_MONTH_COL}
    for item in metadata:
        required_cols.update(item["columns"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True
    total_rows = 0

    for chunk in pd.read_csv(
        data_path,
        usecols=sorted(required_cols),
        chunksize=chunk_size,
        low_memory=False,
    ):
        age_matrix = build_age_matrix(chunk, metadata)
        total_rows += len(age_matrix)
        age_matrix.to_csv(
            output_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False

    print(
        f"Wrote disease trajectory matrix with {len(metadata)} age columns for {total_rows} participants to {output_path}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a sparse disease trajectory matrix containing age at diagnosis per disease.",
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, help="UKB CSV path.")
    parser.add_argument(
        "--variables-path",
        type=Path,
        default=VARIABLES_PATH,
        help="Path to variables.xlsx containing disease report time sheet.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the disease trajectory CSV.",
    )
    parser.add_argument("--chunk-size", type=int, default=5000, help="Chunk size for streaming CSV read.")
    args = parser.parse_args()

    generate_disease_trajectory(
        data_path=args.data_path,
        variables_path=args.variables_path,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
