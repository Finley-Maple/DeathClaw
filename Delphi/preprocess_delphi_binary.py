"""
Convert disease trajectory data to Delphi binary format, using the shared
cohort_split.json for aligned train/val/test splits.

This replaces the interactive notebook (respiratory_preprocess_delphi.ipynb)
and ensures Delphi binary data uses the **same patient splits** as all other
methods (benchmarking, text-embedding, trajectory-embedding).

Input:
    - data/preprocessed/disease_trajectory.csv  (age-at-diagnosis matrix)
    - benchmarking/autoprognosis_survival_dataset.csv  (demographics + survival)
    - Delphi/delphi_labels_chapters_colours_icd.csv  (label index mapping)
    - evaluation/cohort_split.json  (shared train/val/test eids)

Output:
    - Delphi/data/ukb_respiratory_data/train.bin
    - Delphi/data/ukb_respiratory_data/val.bin
    - Delphi/data/ukb_respiratory_data/test.bin

Binary format: uint32 records [patient_id, days_from_birth, label_index]
sorted by patient_id, then by time.

Usage:
    python Delphi/preprocess_delphi_binary.py
    python Delphi/preprocess_delphi_binary.py --output-dir Delphi/data/ukb_respiratory_data
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Defaults
DEFAULT_TRAJECTORY = PROJECT_ROOT / "data" / "preprocessed" / "disease_trajectory.csv"
DEFAULT_SURVIVAL = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
DEFAULT_LABELS = PROJECT_ROOT / "Delphi" / "delphi_labels_chapters_colours_icd.csv"
DEFAULT_COHORT = PROJECT_ROOT / "evaluation" / "cohort_split.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "Delphi" / "data" / "ukb_respiratory_data"
# Optional: raw UKB data for more accurate death dates
DEFAULT_RAW_UKB = PROJECT_ROOT / "data" / "ukb_respiratory_cohort_total.csv"


# ── Label mapping ─────────────────────────────────────────────────────────────

def load_label_mapping(labels_csv: Path) -> dict:
    """
    Build {icd_code -> label_index} and {special_name -> label_index} from
    the Delphi labels CSV.

    The CSV has columns: index, name, count, ...
    Names are like "A00 Cholera", "Female", "BMI low", "Death", etc.
    """
    df = pd.read_csv(labels_csv)
    mapping = {}

    for _, row in df.iterrows():
        idx = int(row["index"])
        name = str(row["name"]).strip()

        # Special tokens (exact names)
        if name in ("Padding", "No event", "Female", "Male",
                     "BMI low", "BMI mid", "BMI high",
                     "Smoking low", "Smoking mid", "Smoking high",
                     "Alcohol low", "Alcohol mid", "Alcohol high",
                     "Death"):
            # Normalize to key
            key = name.replace(" ", "_")
            mapping[key] = idx
            continue

        # ICD codes: first word of name (e.g. "A00" from "A00 Cholera")
        parts = name.split(" ", 1)
        icd_code = parts[0].upper()
        if re.match(r"^[A-Z]\d{2}$", icd_code):
            mapping[icd_code] = idx

    return mapping


def extract_icd_from_column(col_name: str):
    """
    Extract ICD10 code from trajectory column name.
    Column: date_{icd_slug}_first_reported_{name}_{field_id}_age
    Example: date_e78_first_reported_..._130636_age -> 'E78'
    """
    m = re.match(r"^date_([a-z]\d{2})_", col_name, re.IGNORECASE)
    return m.group(1).upper() if m else None


# ── Demographic encoding ─────────────────────────────────────────────────────

def encode_demographics(surv_df: pd.DataFrame, label_map: dict):
    """
    Encode demographics from autoprognosis_survival_dataset.csv into
    (eid, 0, label_index) records at day 0.

    Demographics in survival dataset:
      sex: 0=female, 1=male
      bmi: continuous (<22 low, 22-28 mid, >28 high)
      smoking_status: 0=never(low), 1=previous(mid), 2=current(high)
      alcohol_status: 1=daily(high), 2-3=regular(mid), 4-6=occasional/never(low)
    """
    records = []

    # Sex
    if "sex" in surv_df.columns:
        sex_data = surv_df[["eid", "sex"]].dropna()
        female_mask = sex_data["sex"] == 0
        male_mask = sex_data["sex"] == 1
        if female_mask.sum() > 0 and "Female" in label_map:
            recs = np.column_stack([
                sex_data.loc[female_mask, "eid"].values,
                np.zeros(female_mask.sum(), dtype=int),
                np.full(female_mask.sum(), label_map["Female"], dtype=int),
            ])
            records.append(recs)
        if male_mask.sum() > 0 and "Male" in label_map:
            recs = np.column_stack([
                sex_data.loc[male_mask, "eid"].values,
                np.zeros(male_mask.sum(), dtype=int),
                np.full(male_mask.sum(), label_map["Male"], dtype=int),
            ])
            records.append(recs)

    # BMI
    if "bmi" in surv_df.columns:
        bmi_data = surv_df[["eid", "bmi"]].dropna()
        if len(bmi_data) > 0:
            bmi_vals = bmi_data["bmi"].values
            bmi_labels = np.where(
                bmi_vals > 28, label_map.get("BMI_high", 6),
                np.where(bmi_vals > 22, label_map.get("BMI_mid", 5),
                         label_map.get("BMI_low", 4))
            )
            recs = np.column_stack([
                bmi_data["eid"].values,
                np.zeros(len(bmi_data), dtype=int),
                bmi_labels,
            ])
            records.append(recs)

    # Smoking
    if "smoking_status" in surv_df.columns:
        sm_data = surv_df[["eid", "smoking_status"]].dropna()
        sm_data = sm_data[sm_data["smoking_status"] >= 0]  # Remove invalid
        if len(sm_data) > 0:
            sm_vals = sm_data["smoking_status"].values
            sm_labels = np.where(
                sm_vals == 2, label_map.get("Smoking_high", 9),
                np.where(sm_vals == 1, label_map.get("Smoking_mid", 8),
                         label_map.get("Smoking_low", 7))
            )
            recs = np.column_stack([
                sm_data["eid"].values,
                np.zeros(len(sm_data), dtype=int),
                sm_labels,
            ])
            records.append(recs)

    # Alcohol
    if "alcohol_status" in surv_df.columns:
        alc_data = surv_df[["eid", "alcohol_status"]].dropna()
        alc_data = alc_data[alc_data["alcohol_status"] >= 1]  # Remove invalid
        if len(alc_data) > 0:
            alc_vals = alc_data["alcohol_status"].values
            alc_labels = np.where(
                alc_vals == 1, label_map.get("Alcohol_high", 12),
                np.where(alc_vals <= 3, label_map.get("Alcohol_mid", 11),
                         label_map.get("Alcohol_low", 10))
            )
            recs = np.column_stack([
                alc_data["eid"].values,
                np.zeros(len(alc_data), dtype=int),
                alc_labels,
            ])
            records.append(recs)

    return records


# ── Death events ──────────────────────────────────────────────────────────────

def encode_death_events(surv_df: pd.DataFrame, label_map: dict):
    """
    Encode death events. Uses duration_days (from age 60) + birth year to
    estimate age at death in days from birth.

    duration_days = observation_date - start_date (where start_date ≈ 60th birthday)
    age_at_death_days ≈ 60 * 365.25 + duration_days
    """
    death_label = label_map.get("Death")
    if death_label is None:
        print("  WARNING: 'Death' label not found in mapping, skipping death events.")
        return []

    # Patients who died (event_flag == 1)
    death_data = surv_df[surv_df["event_flag"] == 1][["eid", "duration_days", "age"]].dropna()
    if len(death_data) == 0:
        return []

    # age_at_death_days = 60 years in days + duration_days
    age_at_death_days = (60 * 365.25 + death_data["duration_days"].values).astype(int)

    records = np.column_stack([
        death_data["eid"].values.astype(int),
        age_at_death_days,
        np.full(len(death_data), death_label, dtype=int),
    ])

    return [records]


def encode_death_events_raw(raw_df: pd.DataFrame, label_map: dict):
    """
    Encode death events from raw UKB data (more accurate).
    Uses p40000_i0 (date of death) and p34 (birth year).
    """
    death_label = label_map.get("Death")
    if death_label is None:
        return []

    needed = ["eid", "p40000_i0", "p34"]
    if not all(c in raw_df.columns for c in needed):
        return []

    death_data = raw_df[needed].dropna()
    if len(death_data) == 0:
        return []

    death_dates = pd.to_datetime(death_data["p40000_i0"], errors="coerce")
    birth_years = death_data["p34"].values
    death_days = ((death_dates.dt.year - birth_years) * 365.25 + death_dates.dt.dayofyear).values

    valid = ~pd.isna(death_days) & (death_days > 0)
    if valid.sum() == 0:
        return []

    records = np.column_stack([
        death_data.loc[valid, "eid"].values.astype(int),
        death_days[valid].astype(int),
        np.full(valid.sum(), death_label, dtype=int),
    ])

    return [records]


# ── Main conversion ──────────────────────────────────────────────────────────

def convert_to_delphi_binary(
    trajectory_csv: Path,
    survival_csv: Path,
    labels_csv: Path,
    cohort_json: Path,
    output_dir: Path,
    raw_ukb_csv: Path = None,
):
    """Convert disease trajectory + demographics to Delphi binary format."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load label mapping
    print(f"Loading label mapping from {labels_csv}...")
    label_map = load_label_mapping(labels_csv)
    print(f"  {len(label_map)} labels loaded (incl. {sum(1 for k in label_map if re.match(r'^[A-Z]\\d{2}$', k))} ICD codes)")

    # 2. Load disease trajectory data
    print(f"Loading trajectory data from {trajectory_csv}...")
    traj_df = pd.read_csv(trajectory_csv, low_memory=False)
    traj_df["eid"] = traj_df["eid"].astype(int)
    print(f"  {len(traj_df)} patients, {len(traj_df.columns)} columns")

    # Identify age columns
    age_columns = [c for c in traj_df.columns if c.endswith("_age") and c != "eid"]
    print(f"  {len(age_columns)} disease age columns found")

    # 3. Convert disease events to records
    print("Converting disease events...")
    data_list = []
    mapped_count = 0
    unmapped_count = 0

    for col in tqdm(age_columns, desc="Disease columns"):
        icd_code = extract_icd_from_column(col)
        if icd_code is None or icd_code not in label_map:
            unmapped_count += 1
            continue

        label_idx = label_map[icd_code]
        mask = traj_df[col].notna()
        if mask.sum() == 0:
            continue

        patient_ids = traj_df.loc[mask, "eid"].values.astype(int)
        age_years = traj_df.loc[mask, col].values.astype(float)
        age_days = (age_years * 365.25).astype(int)

        # Filter valid ages (>= 0)
        valid = age_days >= 0
        if valid.sum() == 0:
            continue

        records = np.column_stack([
            patient_ids[valid],
            age_days[valid],
            np.full(valid.sum(), label_idx, dtype=int),
        ])
        data_list.append(records)
        mapped_count += 1

    print(f"  Mapped {mapped_count} disease columns, {unmapped_count} unmapped")

    # 4. Load survival data for demographics + death
    print(f"Loading survival data from {survival_csv}...")
    surv_df = pd.read_csv(survival_csv, low_memory=False)
    surv_df["eid"] = surv_df["eid"].astype(int)

    # Add demographics
    print("Encoding demographics...")
    demo_records = encode_demographics(surv_df, label_map)
    data_list.extend(demo_records)
    demo_count = sum(len(r) for r in demo_records)
    print(f"  Added {demo_count} demographic records")

    # Add death events
    print("Encoding death events...")
    if raw_ukb_csv and raw_ukb_csv.exists():
        print(f"  Using raw UKB data: {raw_ukb_csv}")
        raw_cols = ["eid", "p40000_i0", "p34"]
        raw_df = pd.read_csv(raw_ukb_csv, usecols=raw_cols, low_memory=False)
        death_records = encode_death_events_raw(raw_df, label_map)
    else:
        print("  Using survival dataset (approximate age at death)")
        death_records = encode_death_events(surv_df, label_map)
    data_list.extend(death_records)
    death_count = sum(len(r) for r in death_records)
    print(f"  Added {death_count} death records")

    # 5. Combine all records
    if not data_list:
        raise ValueError("No records created! Check your data and label mapping.")

    all_data = np.vstack(data_list).astype(np.int64)
    print(f"\nTotal records: {len(all_data)}")

    # Sort by patient_id, then by time
    sort_idx = np.lexsort((all_data[:, 1], all_data[:, 0]))
    all_data = all_data[sort_idx]

    # Remove duplicates (same patient, same label)
    df_dedup = pd.DataFrame(all_data, columns=["patient_id", "days", "label"])
    df_dedup = df_dedup.drop_duplicates(subset=["patient_id", "label"])
    all_data = df_dedup.values.astype(np.int64)
    print(f"Records after deduplication: {len(all_data)}")

    # 6. Load cohort split
    print(f"\nLoading cohort split from {cohort_json}...")
    with open(cohort_json) as f:
        cohort = json.load(f)
    train_eids = set(cohort["train_eids"])
    val_eids = set(cohort["val_eids"])
    test_eids = set(cohort["test_eids"])
    print(f"  Train: {len(train_eids)}, Val: {len(val_eids)}, Test: {len(test_eids)}")

    # 7. Split data according to cohort
    patient_ids = all_data[:, 0]
    train_mask = np.isin(patient_ids, list(train_eids))
    val_mask = np.isin(patient_ids, list(val_eids))
    test_mask = np.isin(patient_ids, list(test_eids))

    train_data = all_data[train_mask].astype(np.uint32)
    val_data = all_data[val_mask].astype(np.uint32)
    test_data = all_data[test_mask].astype(np.uint32)

    # Count unique patients in each split
    train_patients = len(set(train_data[:, 0])) if len(train_data) > 0 else 0
    val_patients = len(set(val_data[:, 0])) if len(val_data) > 0 else 0
    test_patients = len(set(test_data[:, 0])) if len(test_data) > 0 else 0

    # Also save full dataset (for Delphi training if needed)
    full_data = all_data.astype(np.uint32)

    # 8. Save binary files
    full_path = output_dir / "full.bin"
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    test_path = output_dir / "test.bin"

    full_data.tofile(str(full_path))
    train_data.tofile(str(train_path))
    val_data.tofile(str(val_path))
    test_data.tofile(str(test_path))

    print(f"\n=== Output ===")
    print(f"  Full:  {full_path} ({len(full_data)} records)")
    print(f"  Train: {train_path} ({len(train_data)} records, {train_patients} patients)")
    print(f"  Val:   {val_path} ({len(val_data)} records, {val_patients} patients)")
    print(f"  Test:  {test_path} ({len(test_data)} records, {test_patients} patients)")

    # 9. Save metadata
    metadata = {
        "total_records": len(full_data),
        "total_patients": len(set(full_data[:, 0])),
        "train_records": len(train_data),
        "train_patients": train_patients,
        "val_records": len(val_data),
        "val_patients": val_patients,
        "test_records": len(test_data),
        "test_patients": test_patients,
        "disease_columns_mapped": mapped_count,
        "disease_columns_unmapped": unmapped_count,
        "demographic_records": demo_count,
        "death_records": death_count,
        "labels_used": len(set(all_data[:, 2])),
    }
    meta_path = output_dir / "preprocessing_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata: {meta_path}")

    # Verification
    print(f"\n=== Verification ===")
    if len(train_data) > 0:
        sample_pid = train_data[0, 0]
        sample_records = train_data[train_data[:, 0] == sample_pid]
        print(f"  Sample patient {sample_pid}: {len(sample_records)} records")
        for r in sample_records[:8]:
            age_years = r[1] / 365.25
            print(f"    Day {r[1]:6d} (age {age_years:5.1f}): label {r[2]}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Convert disease trajectory to Delphi binary format "
                    "(aligned with shared cohort split)."
    )
    parser.add_argument("--trajectory-csv", type=Path, default=DEFAULT_TRAJECTORY,
                        help="Path to disease_trajectory.csv")
    parser.add_argument("--survival-csv", type=Path, default=DEFAULT_SURVIVAL,
                        help="Path to autoprognosis_survival_dataset.csv")
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS,
                        help="Path to delphi_labels_chapters_colours_icd.csv")
    parser.add_argument("--cohort-json", type=Path, default=DEFAULT_COHORT,
                        help="Path to cohort_split.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory for .bin files")
    parser.add_argument("--raw-ukb-csv", type=Path, default=DEFAULT_RAW_UKB,
                        help="Optional: raw UKB CSV for more accurate death dates")
    args = parser.parse_args()

    # Validate inputs
    for name, path in [("trajectory", args.trajectory_csv),
                       ("survival", args.survival_csv),
                       ("labels", args.labels_csv),
                       ("cohort", args.cohort_json)]:
        if not path.exists():
            print(f"ERROR: {name} file not found: {path}")
            sys.exit(1)

    raw_ukb = args.raw_ukb_csv if args.raw_ukb_csv.exists() else None
    if raw_ukb:
        print(f"Using raw UKB data for death dates: {raw_ukb}")
    else:
        print(f"Raw UKB data not found at {args.raw_ukb_csv}, using survival dataset")

    convert_to_delphi_binary(
        trajectory_csv=args.trajectory_csv,
        survival_csv=args.survival_csv,
        labels_csv=args.labels_csv,
        cohort_json=args.cohort_json,
        output_dir=args.output_dir,
        raw_ukb_csv=raw_ukb,
    )


if __name__ == "__main__":
    main()
