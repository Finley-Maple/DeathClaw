"""
Generate binary disease features based on ICD10 codes diagnosed before age 60.

This script uses ICD10_FLAG_DEFINITIONS to create disease flags by:
1. Using UK Biobank's "First Occurrence" fields (p130xxx-p132xxx) which contain dates of first ICD10 diagnosis
2. Checking if the diagnosis date is before the participant's 60th birthday
3. Matching ICD10 codes to their corresponding field IDs via the icd10_codes_mod.tsv mapping file

UK Biobank First Occurrence fields:
- Even field IDs (e.g., p130706) = Source of report for the ICD10 code
- Odd field IDs (e.g., p130707) = Date of first diagnosis for the ICD10 code
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import pandas as pd

from preprocess_survival import (
    DATA_PATH,
    PROJECT_ROOT,
    build_start_dates,
    load_header,
)

# ICD10 flag definitions: each disease maps to ICD10 prefixes and/or ranges
ICD10_FLAG_DEFINITIONS = {
    "atrial_fibrillation": {"prefixes": ["I48"]},
    "diabetes_t1": {"prefixes": ["E10"]},
    "diabetes_t2": {"prefixes": ["E11"]},
    "diabetes_any": {"ranges": [("E10", "E14")]},
    "rheumatoid_arthritis": {"ranges": [("M05", "M06")]},
    "ckd": {"ranges": [("N17", "N19")]},
    "migraine": {"prefixes": ["G43"]},
    "systemic_lupus": {"prefixes": ["M32"]},
    "hiv": {"ranges": [("B20", "B24")]},
    "mental_illness": {"ranges": [("F20", "F25"), ("F28", "F31")]},
    "hypertension_diagnosis": {"ranges": [("I10", "I15")]},
    "cholesterol_disorder": {"prefixes": ["E78"]},
    "depression": {"ranges": [("F32", "F33")]},
    "stroke": {"ranges": [("I60", "I64")], "prefixes": ["G45"]},
    "erectile_dysfunction": {"prefixes": ["N48", "F52"]},
}

# Path to the ICD10 code to field ID mapping file
ICD10_MAPPING_PATH = PROJECT_ROOT / "Delphi" / "data" / "ukb_simulated_data" / "icd10_codes_mod.tsv"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarking" / "disease_before60_features.csv"


def load_icd10_field_mapping(mapping_path: Path) -> Dict[str, int]:
    """
    Load mapping from ICD10 codes to UK Biobank field IDs.
    
    The mapping file contains lines like:
    f.130706.0.0    Source of report of E10 (insulin-dependent diabetes mellitus)
    
    This extracts the field ID (130706) and the ICD10 code (E10).
    The date field is always field_id + 1 (e.g., 130707 for E10).
    
    Returns:
        Dict mapping ICD10 code (e.g., "E10") to date field ID (e.g., 130707)
    """
    icd10_to_field: Dict[str, int] = {}
    
    if not mapping_path.exists():
        print(f"Warning: ICD10 mapping file not found at {mapping_path}")
        return icd10_to_field
    
    with mapping_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            
            field_part, description = parts
            
            # Extract field ID from format like "f.130706.0.0"
            match = re.match(r"f\.(\d+)\.", field_part)
            if not match:
                continue
            field_id = int(match.group(1))
            
            # Extract ICD10 code from description like "Source of report of E10 (...)"
            icd_match = re.search(r"Source of report of ([A-Z]\d+)", description)
            if not icd_match:
                continue
            icd10_code = icd_match.group(1)
            
            # The date field is always source_field_id + 1
            date_field_id = field_id + 1
            icd10_to_field[icd10_code] = date_field_id
    
    print(f"Loaded {len(icd10_to_field)} ICD10 code to field ID mappings")
    return icd10_to_field


def expand_icd10_range(start: str, end: str) -> List[str]:
    """
    Expand an ICD10 range like ("E10", "E14") to ["E10", "E11", "E12", "E13", "E14"].
    
    Handles both letter+number format (E10-E14) and pure number increments.
    """
    if not start or not end:
        return []
    
    # Extract letter prefix and numeric part
    start_match = re.match(r"([A-Z]+)(\d+)", start)
    end_match = re.match(r"([A-Z]+)(\d+)", end)
    
    if not start_match or not end_match:
        return [start]
    
    start_letter, start_num = start_match.groups()
    end_letter, end_num = end_match.groups()
    
    # Only expand if same letter prefix
    if start_letter != end_letter:
        return [start, end]
    
    codes = []
    for num in range(int(start_num), int(end_num) + 1):
        codes.append(f"{start_letter}{num}")
    
    return codes


def get_icd10_codes_for_disease(definition: Dict) -> List[str]:
    """
    Get all ICD10 codes for a disease definition.
    
    A definition can have:
    - "prefixes": List of exact ICD10 codes (e.g., ["I48", "E10"])
    - "ranges": List of (start, end) tuples to expand (e.g., [("E10", "E14")])
    """
    codes = []
    
    # Add exact prefix matches
    if "prefixes" in definition:
        codes.extend(definition["prefixes"])
    
    # Expand ranges
    if "ranges" in definition:
        for start, end in definition["ranges"]:
            codes.extend(expand_icd10_range(start, end))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_codes = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)
    
    return unique_codes


def build_disease_field_mapping(
    icd10_to_field: Dict[str, int],
    header: Sequence[str],
) -> Dict[str, List[str]]:
    """
    Build mapping from disease names to available date field columns.
    
    For each disease in ICD10_FLAG_DEFINITIONS:
    1. Get all relevant ICD10 codes
    2. Map each code to its date field ID
    3. Check if the date field column exists in the dataset header
    
    Returns:
        Dict mapping disease name to list of date column names (e.g., ["p130707"])
    """
    header_set = set(header)
    disease_to_columns: Dict[str, List[str]] = {}
    
    for disease_name, definition in ICD10_FLAG_DEFINITIONS.items():
        icd10_codes = get_icd10_codes_for_disease(definition)
        columns = []
        
        for code in icd10_codes:
            if code in icd10_to_field:
                date_field_id = icd10_to_field[code]
                col_name = f"p{date_field_id}"
                if col_name in header_set:
                    columns.append(col_name)
        
        disease_to_columns[disease_name] = columns
        
        if columns:
            print(f"  {disease_name}: {len(columns)} date columns found ({len(icd10_codes)} ICD10 codes)")
        else:
            print(f"  {disease_name}: No date columns found ({len(icd10_codes)} ICD10 codes)")
    
    return disease_to_columns


def build_pre60_flags(
    chunk: pd.DataFrame,
    disease_to_columns: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Build binary disease flags for each participant based on diagnosis dates before age 60.
    
    For each disease:
    1. Get all date columns associated with its ICD10 codes
    2. Find the earliest diagnosis date across all columns
    3. Check if diagnosis date < 60th birthday (start_date)
    4. Set flag to 1 if diagnosed before 60, 0 otherwise
    """
    chunk = chunk.copy()
    chunk["eid"] = chunk["eid"].astype(int)
    chunk["start_date"] = build_start_dates(chunk)
    
    chunk_features = pd.DataFrame({"eid": chunk["eid"]})
    date_cache: Dict[str, pd.Series] = {}
    
    for disease_name, cols in disease_to_columns.items():
        if not cols:
            # No columns available for this disease
            chunk_features[disease_name] = 0
            continue
        
        # Parse dates for each column (with caching)
        date_series: List[pd.Series] = []
        for col in cols:
            if col not in date_cache:
                if col in chunk.columns:
                    date_cache[col] = pd.to_datetime(chunk[col], errors="coerce")
                else:
                    date_cache[col] = pd.Series(pd.NaT, index=chunk.index)
            date_series.append(date_cache[col])
        
        # Find earliest diagnosis date across all ICD10 codes for this disease
        stacked = pd.concat(date_series, axis=1)
        earliest_diagnosis = stacked.min(axis=1)
        
        # Flag is 1 if diagnosis date exists and is before 60th birthday
        start = chunk["start_date"]
        flag = (
            earliest_diagnosis.notna()
            & start.notna()
            & (earliest_diagnosis < start)
        ).astype(int)
        
        chunk_features[disease_name] = flag
    
    return chunk_features


def preprocess_disease_features(
    data_path: Path,
    output_path: Path,
    icd10_mapping_path: Path,
    chunk_size: int = 5000,
) -> None:
    """
    Main preprocessing function to extract disease diagnosis features.
    
    Steps:
    1. Load ICD10 code to field ID mapping
    2. Load dataset header to find available columns
    3. Build mapping from disease names to date columns
    4. Process data in chunks, creating binary flags for each disease
    5. Write results to output CSV
    """
    print("Loading ICD10 code to field ID mapping...")
    icd10_to_field = load_icd10_field_mapping(icd10_mapping_path)
    
    print("Loading dataset header...")
    header = load_header(data_path)
    
    print("\nBuilding disease to column mapping...")
    disease_to_columns = build_disease_field_mapping(icd10_to_field, header)
    
    # Count diseases with available columns
    diseases_with_data = sum(1 for cols in disease_to_columns.values() if cols)
    if diseases_with_data == 0:
        print("\nWarning: No ICD10 first occurrence fields found in dataset.")
        print("The disease features will all be 0.")
    
    # Collect all needed columns
    usecols: Set[str] = {"eid", "p34", "p52"}
    for cols in disease_to_columns.values():
        usecols.update(cols)
    
    # Filter to only columns that exist in header
    header_set = set(header)
    usecols = usecols.intersection(header_set)
    
    print(f"\nProcessing {len(usecols)} columns for {len(disease_to_columns)} diseases...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True
    total_rows = 0
    
    for chunk in pd.read_csv(
        data_path,
        usecols=sorted(usecols),
        chunksize=chunk_size,
        low_memory=False,
    ):
        chunk_features = build_pre60_flags(chunk, disease_to_columns)
        total_rows += len(chunk_features)
        chunk_features.to_csv(
            output_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False
        
        if total_rows % 50000 == 0:
            print(f"  Processed {total_rows:,} rows...")
    
    print(f"\nWrote {total_rows:,} participant rows with {len(disease_to_columns)} disease features to {output_path}")


def print_disease_summary(output_path: Path) -> None:
    """Print summary statistics for the generated disease features."""
    print("\n" + "=" * 60)
    print("DISEASE FEATURE SUMMARY")
    print("=" * 60)
    
    try:
        # Read a sample to get column names and counts
        df = pd.read_csv(output_path, nrows=None)
        disease_cols = [c for c in df.columns if c != "eid"]
        
        print(f"\nTotal participants: {len(df):,}")
        print(f"Total disease features: {len(disease_cols)}")
        print("\nPrevalence (diagnosed before age 60):")
        print("-" * 40)
        
        for col in sorted(disease_cols):
            count = df[col].sum()
            pct = count / len(df) * 100
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    except Exception as e:
        print(f"Could not generate summary: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create binary features for diseases diagnosed before age 60 using ICD10 codes.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to the UKB CSV file.",
    )
    parser.add_argument(
        "--icd10-mapping-path",
        type=Path,
        default=ICD10_MAPPING_PATH,
        help="Path to icd10_codes_mod.tsv mapping file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the engineered feature CSV.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of rows to process per chunk.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics after processing.",
    )
    args = parser.parse_args()

    preprocess_disease_features(
        data_path=args.data_path,
        output_path=args.output_path,
        icd10_mapping_path=args.icd10_mapping_path,
        chunk_size=args.chunk_size,
    )
    
    if args.summary:
        print_disease_summary(args.output_path)


if __name__ == "__main__":
    main()
