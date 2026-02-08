"""
Generate Delphi-style trajectory text per patient, truncated at age 60.

For each patient in the shared cohort, produces a text like:
    0.0: Male
    2.0: B01 Varicella [chickenpox]
    5.0: No event
    10.0: No event
    20.0: G43 Migraine
    ...
    55.0: No event

Events after age 60 are excluded. The output is keyed by eid.

Usage:
    python preprocessing/generate_trajectory_text.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default paths
DISEASE_TRAJECTORY_CSV = PROJECT_ROOT / "data" / "preprocessed" / "disease_trajectory.csv"
SURVIVAL_DATASET = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
COHORT_SPLIT_JSON = PROJECT_ROOT / "evaluation" / "cohort_split.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed" / "trajectory_before60"
OUTPUT_CSV = PROJECT_ROOT / "data" / "preprocessed" / "trajectory_before60.csv"

# Age cutoff
AGE_CUTOFF = 60.0

# "No event" ages: every 5 years from 0 to 55
NO_EVENT_AGES = list(range(0, 60, 5))


def load_sex_mapping(survival_csv: Path) -> Dict[int, str]:
    """Load sex from survival dataset. sex: 0=Female, 1=Male."""
    df = pd.read_csv(survival_csv, usecols=["eid", "sex"])
    df["eid"] = df["eid"].astype(int)
    sex_map = {}
    for _, row in df.iterrows():
        val = row["sex"]
        if pd.notna(val):
            sex_map[int(row["eid"])] = "Male" if int(val) == 1 else "Female"
    return sex_map


def load_trajectory_matrix(trajectory_csv: Path, eids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load the disease trajectory matrix (age at diagnosis per disease per patient).
    Columns: eid, <disease_name>_<field_id>_age, ...
    """
    df = pd.read_csv(trajectory_csv)
    df["eid"] = df["eid"].astype(int)
    if eids is not None:
        df = df[df["eid"].isin(set(eids))]
    return df


def build_trajectory_text(
    row: pd.Series,
    age_columns: List[str],
    sex: Optional[str] = None,
    age_cutoff: float = AGE_CUTOFF,
) -> str:
    """
    Build a Delphi-style trajectory string for one patient.

    Events are sorted by age. "No event" markers are inserted at 5-year
    intervals where no disease was diagnosed.

    Args:
        row: One row from the trajectory DataFrame.
        age_columns: Column names for disease ages (ending in _age).
        sex: "Male" or "Female" (inserted at age 0.0 if provided).
        age_cutoff: Maximum age to include (exclusive for events, inclusive for no-event).

    Returns:
        Trajectory text string.
    """
    # Collect (age, event_name) pairs
    events: List[Tuple[float, str]] = []

    # Add sex at age 0
    if sex:
        events.append((0.0, sex))

    # Add disease events before cutoff
    for col in age_columns:
        age_val = row.get(col)
        if pd.notna(age_val):
            age_years = float(age_val)
            if 0 <= age_years < age_cutoff:
                # Derive a readable disease name from the column
                # Column format: slug_name_fieldid_age
                # Remove the trailing _fieldid_age to get the disease slug
                parts = col.rsplit("_", 2)
                if len(parts) >= 3:
                    disease_name = parts[0].replace("_", " ").title()
                else:
                    disease_name = col.replace("_", " ").title()
                events.append((age_years, disease_name))

    # Sort events by age
    events.sort(key=lambda x: x[0])

    # Determine which 5-year marks have no event within ±2.5 years
    event_ages = set(round(e[0]) for e in events if e[1] not in ("Male", "Female"))
    no_event_markers: List[Tuple[float, str]] = []
    for age in NO_EVENT_AGES:
        if age >= age_cutoff:
            break
        # Check if any event falls in [age-2.5, age+2.5)
        has_event = any(
            abs(e[0] - age) < 2.5
            for e in events
            if e[1] not in ("Male", "Female")
        )
        if not has_event:
            no_event_markers.append((float(age), "No event"))

    all_events = events + no_event_markers
    all_events.sort(key=lambda x: (x[0], x[1] == "No event"))

    # Build text lines
    lines = []
    for age, event_name in all_events:
        lines.append(f"{age:.1f}: {event_name}")

    return "\n".join(lines) if lines else "No events recorded before age 60."


def generate_trajectory_texts(
    trajectory_csv: Path = DISEASE_TRAJECTORY_CSV,
    survival_csv: Path = SURVIVAL_DATASET,
    cohort_json: Path = COHORT_SPLIT_JSON,
    output_dir: Optional[Path] = OUTPUT_DIR,
    output_csv: Optional[Path] = OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Generate trajectory texts for the shared cohort.

    Returns DataFrame with columns [eid, trajectory_text].
    """
    # Load cohort eids
    eids = None
    if cohort_json.exists():
        with open(cohort_json) as f:
            cohort = json.load(f)
        eids = cohort["train_eids"] + cohort["val_eids"] + cohort["test_eids"]
        print(f"Loaded {len(eids)} eids from cohort split.")

    # Load sex mapping
    sex_map = load_sex_mapping(survival_csv)
    print(f"Loaded sex for {len(sex_map)} participants.")

    # Load trajectory matrix
    if not trajectory_csv.exists():
        print(f"Warning: Trajectory CSV not found at {trajectory_csv}")
        print("Run preprocessing/generate_disease_trajectory.py first.")
        print("Falling back to survival dataset disease flags...")
        return _fallback_from_survival(survival_csv, eids, sex_map, output_dir, output_csv)

    traj_df = load_trajectory_matrix(trajectory_csv, eids)
    print(f"Loaded trajectory matrix: {len(traj_df)} patients, {len(traj_df.columns) - 1} disease columns.")

    # Identify age columns (those ending with _age)
    age_columns = [c for c in traj_df.columns if c.endswith("_age") and c != "eid"]

    records: List[Dict] = []
    for _, row in traj_df.iterrows():
        eid = int(row["eid"])
        sex = sex_map.get(eid)
        text = build_trajectory_text(row, age_columns, sex=sex)
        records.append({"eid": eid, "trajectory_text": text})

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"eid_{eid}.txt").write_text(text, encoding="utf-8")

    result_df = pd.DataFrame(records)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"Wrote {len(result_df)} trajectory texts to {output_csv}")

    return result_df


def _fallback_from_survival(
    survival_csv: Path,
    eids: Optional[List[int]],
    sex_map: Dict[int, str],
    output_dir: Optional[Path],
    output_csv: Optional[Path],
) -> pd.DataFrame:
    """
    Fallback: generate simple trajectory text from the binary disease flags
    in the survival dataset (without exact ages, uses age 0 for sex, no
    precise event ages).
    """
    surv_df = pd.read_csv(survival_csv)
    surv_df["eid"] = surv_df["eid"].astype(int)
    if eids is not None:
        surv_df = surv_df[surv_df["eid"].isin(set(eids))]

    # Disease flag columns from preprocess_diagnosis.py
    from benchmarking.preprocess_diagnosis import ICD10_FLAG_DEFINITIONS
    disease_cols = [c for c in ICD10_FLAG_DEFINITIONS.keys() if c in surv_df.columns]

    DISEASE_NAMES_READABLE = {
        "atrial_fibrillation": "I48 Atrial fibrillation",
        "diabetes_t1": "E10 Type 1 diabetes",
        "diabetes_t2": "E11 Type 2 diabetes",
        "diabetes_any": "E10-E14 Diabetes",
        "ckd": "N17-N19 Chronic kidney disease",
        "migraine": "G43 Migraine",
        "systemic_lupus": "M32 Systemic lupus erythematosus",
        "hiv": "B20-B24 HIV",
        "mental_illness": "F20-F31 Severe mental illness",
        "lipid_disorder": "E78 Lipid disorder",
        "cholesterol_disorder": "E78 Cholesterol disorder",
        "depression": "F32-F33 Depression",
        "stroke": "I60-I64/G45 Stroke/TIA",
        "erectile_dysfunction": "N48/F52 Erectile dysfunction",
        "rheumatoid_arthritis": "M05-M06 Rheumatoid arthritis",
        "hypertension_diagnosis": "I10-I15 Hypertension",
    }

    records: List[Dict] = []
    for _, row in surv_df.iterrows():
        eid = int(row["eid"])
        sex = sex_map.get(eid)
        events: List[Tuple[float, str]] = []

        if sex:
            events.append((0.0, sex))

        # Add diseases (age unknown, use placeholder text)
        for col in disease_cols:
            if pd.notna(row.get(col)) and int(row[col]) == 1:
                name = DISEASE_NAMES_READABLE.get(col, col.replace("_", " ").title())
                events.append((None, name))

        # Build text
        lines = []
        for age, name in events:
            if age is not None:
                lines.append(f"{age:.1f}: {name}")
            else:
                lines.append(f"Before 60: {name}")

        text = "\n".join(lines) if lines else "No events recorded before age 60."
        records.append({"eid": eid, "trajectory_text": text})

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"eid_{eid}.txt").write_text(text, encoding="utf-8")

    result_df = pd.DataFrame(records)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False)
        print(f"Wrote {len(result_df)} trajectory texts (fallback) to {output_csv}")
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Delphi-style trajectory text per patient, truncated at age 60."
    )
    parser.add_argument(
        "--trajectory-csv", type=Path, default=DISEASE_TRAJECTORY_CSV,
        help="Path to disease_trajectory.csv (age-at-diagnosis matrix).",
    )
    parser.add_argument(
        "--survival-csv", type=Path, default=SURVIVAL_DATASET,
        help="Path to autoprognosis_survival_dataset.csv (for sex info).",
    )
    parser.add_argument(
        "--cohort-json", type=Path, default=COHORT_SPLIT_JSON,
        help="Path to cohort_split.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for individual eid_*.txt files.",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=OUTPUT_CSV,
        help="CSV output path (eid, trajectory_text).",
    )
    args = parser.parse_args()

    generate_trajectory_texts(
        trajectory_csv=args.trajectory_csv,
        survival_csv=args.survival_csv,
        cohort_json=args.cohort_json,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
