import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "ukb_respiratory_cohort_total.csv"
COMPARATOR_PATH = PROJECT_ROOT / "Delphi" / "Used_UKB_fields.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "benchmarking"
OUTPUT_DATA = OUTPUT_DIR / "autoprognosis_survival_dataset.csv"
OUTPUT_METADATA = OUTPUT_DIR / "autoprognosis_survival_metadata.json"
DISEASE_FEATURES_PATH = OUTPUT_DIR / "disease_before60_features.csv"

BASE_COLUMNS = ["eid", "p34", "p52", "p40000_i0", "p40020_i0"]
CENSORING_MAP = {
    "E/W": pd.Timestamp("2024-08-31"),
    "E": pd.Timestamp("2024-08-31"),
    "ENGLAND": pd.Timestamp("2024-08-31"),
    "SCOT": pd.Timestamp("2024-11-30"),
    "SCOTLAND": pd.Timestamp("2024-11-30"),
}
MONTH_FALLBACK = 7  # default to July if month-of-birth missing
DEFAULT_CENSOR_DATE = pd.Timestamp("2024-08-31")
EXCLUDED_FIELDS: List[str] = []
TOWNSEND_THRESHOLD = 0.758
LIVING_ALONE_VALUE = 1
MEDICATION_KEYWORDS = {
    "hypertension_meds": [
        "enalapril",
        "lisinopril",
        "perindopril",
        "ramipril",
        "candesartan",
        "irbesartan",
        "losartan",
        "valsartan",
        "olmesartan",
        "amlodipine",
        "felodipine",
        "nifedipine",
        "diltiazem",
        "verapamil",
        "indapamide",
        "bendroflumethiazide",
        "atenolol",
        "bisoprolol",
        "doxazosin",
        "amiloride",
        "spironolactone",
    ],
    "aspirin": ["aspirin"],
    "antipsychotic": [
        "amisulpride",
        "aripiprazole",
        "clozapine",
        "olanzapine",
        "paliperidone",
        "quetiapine",
        "risperidone",
        "sertindole",
        "zotepine",
    ],
    "steroids": [
        "prednisolone",
        "betamethasone",
        "cortisone",
        "dexamethasone",
        "deflazacort",
        "hydrocortisone",
        "methylprednisolone",
        "triamcinolone",
    ],
    "statins": [
        "atorvastatin",
        "simvastatin",
        "rosuvastatin",
        "pravastatin",
        "lovastatin",
        "fluvastatin",
        "pitavastatin",
        "lipitor",
        "lescol",
        "lipostat",
        "crestor",
        "zocor",
        "statin",
    ],
}
ERECTILE_DYSFUNCTION_MEDS = [
    "sildenafil",
    "viagra",
    "tadalafil",
    "cialis",
    "vardenafil",
    "levitra",
    "avanafil",
    "alprostadil",
    "spedra",
    "stendra",
    "papaverine",
    "phentolamine",
]

ENGINEERING_FIELD_IDS = {
    "31",  # sex
    "34",  # age / birth year
    "52",  # month of birth, already in BASE but keep for safety
    "1558",  # alcohol
    "189",  # townsend
    "4079",  # diastolic (not used yet but helpful)
    "4080",  # systolic
    "709",  # living alone
    "20003",  # medications
    "20116",  # smoking
    "21001",  # BMI direct
    "21002",  # weight
    "50",  # height
    "21022",  # age at recruitment (optional)
    "30600",  # albumin
    "30610",  # alkaline phosphatase
    "30620",  # alt
    "30630",  # apolipoprotein A
    "30640",  # apolipoprotein B
    "30670",  # urea
    "30690",  # total cholesterol
    "30700",  # creatinine
    "30720",  # cystatin c
    "30730",  # crp
    "30732",  # ggt
    "30740",  # glucose
    "30750",  # hba1c
    "30760",  # hdl
    "30770",  # igf1
    "30790",  # lipoprotein a
    "30840",  # bilirubin
    "30860",  # total protein
    "30870",  # triglycerides
    "30020",  # haemoglobin
    "30040",  # mcv
}
BASELINE_FIELD_MAP = {
    "sex": "31",
    "age": "34",
    "smoking_status": "20116",
    "alcohol_status": "1558",
    "townsend_index": "189",
}
BIOMARKER_FIELD_MAP = {
    "hdl_cholesterol": "30760",
    "total_cholesterol": "30690",
    "hba1c": "30750",
    "alkaline_phosphatase": "30610",
    "apolipoprotein_a": "30630",
    "apolipoprotein_b": "30640",
    "cystatin_c": "30720",
    "c_reactive_protein": "30730",
    "igf1": "30770",
    "lipoprotein_a": "30790",
    "triglycerides": "30870",
    "creatinine": "30700",
    "urea": "30670",
    "glucose": "30740",
    "albumin": "30600",
    "total_protein": "30860",
    "bilirubin": "30840",
    "gamma_gt": "30732",
    "alt": "30620",
    "haemoglobin": "30020",
    "mcv": "30040",
}


def load_header(path: Path) -> Sequence[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def collect_comparator_codes() -> List[str]:
    comparator_df = pd.read_excel(COMPARATOR_PATH, sheet_name="Comparators")
    codes = (
        pd.to_numeric(comparator_df["UKB Code"], errors="coerce")
        .dropna()
        .astype(int)
        .astype(str)
        .unique()
        .tolist()
    )
    codes = [code for code in codes if code not in EXCLUDED_FIELDS]
    return sorted(codes)


def find_columns_for_code(header: Sequence[str], field_id: str) -> List[str]:
    prefixes = [f"p{field_id}", f"p1{field_id}"]
    for prefix in prefixes:
        matches = [col for col in header if col.startswith(prefix)]
        if matches:
            return matches
    return []


def gather_feature_columns(header: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    codes = collect_comparator_codes()
    feature_cols: List[str] = []
    field_map: Dict[str, List[str]] = {}
    missing: List[str] = []

    for code in codes:
        cols = find_columns_for_code(header, code)
        if cols:
            field_map[code] = cols
            feature_cols.extend(cols)
        else:
            missing.append(code)
    seen = set()
    ordered_features: List[str] = []
    for col in header:
        if col in feature_cols and col not in seen:
            ordered_features.append(col)
            seen.add(col)
    return ordered_features, field_map, missing


def extend_feature_columns(
    header: Sequence[str],
    field_map: Dict[str, List[str]],
    feature_cols: List[str],
    required_field_ids: Set[str],
) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    feature_set = set(feature_cols)

    for code in sorted(required_field_ids):
        if code not in field_map:
            cols = find_columns_for_code(header, code)
            if cols:
                field_map[code] = cols
            else:
                missing.append(code)
                continue
        cols = field_map.get(code, [])
        for col in cols:
            if col not in feature_set:
                feature_cols.append(col)
                feature_set.add(col)

    ordered_features: List[str] = []
    seen: Set[str] = set()
    for col in header:
        if col in feature_set and col not in seen:
            ordered_features.append(col)
            seen.add(col)
    return ordered_features, missing


def get_field_columns_from_map(field_map: Dict[str, List[str]], header: Sequence[str], field_id: str) -> List[str]:
    if field_id in field_map:
        return field_map[field_id]
    cols = find_columns_for_code(header, field_id)
    if cols:
        field_map[field_id] = cols
    return cols


def first_available_value(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    subset = df[available]
    stacked = subset.stack(future_stack=True)
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    if not stacked.empty:
        first_values = stacked.groupby(level=0).first()
        result.loc[first_values.index] = first_values.values
    return result


def medication_keyword_flag(df: pd.DataFrame, columns: Sequence[str], keywords: Sequence[str]) -> pd.Series:
    available = [col for col in columns if col in df.columns]
    if not available or not keywords:
        return pd.Series(0, index=df.index, dtype=int)
    pattern = "|".join(re.escape(keyword.lower()) for keyword in keywords if keyword)
    if not pattern:
        return pd.Series(0, index=df.index, dtype=int)
    lowered = pd.DataFrame(
        {col: df[col].fillna("").astype(str).str.lower() for col in available},
        index=df.index,
    )
    matches = lowered.apply(lambda col: col.str.contains(pattern, regex=True, na=False))
    return matches.any(axis=1).astype(int)


def load_disease_features(eids: Sequence[int], path: Path = DISEASE_FEATURES_PATH) -> pd.DataFrame:
    """
    Load pre-computed disease-before-60 binary features for given participant IDs.
    
    Args:
        eids: Sequence of participant IDs to filter
        path: Path to the disease_before60_features.csv file
    
    Returns:
        DataFrame with eid and disease flag columns for the specified participants
    """
    if not path.exists():
        print(f"Warning: Disease features file not found at {path}")
        print("Run preprocess_diagnosis.py first to generate disease features.")
        return pd.DataFrame({"eid": eids})
    
    ids_set = set(map(int, eids))
    chunks = []
    for chunk in pd.read_csv(path, chunksize=5000, low_memory=False):
        chunk["eid"] = chunk["eid"].astype(int)
        mask = chunk["eid"].isin(ids_set)
        if mask.any():
            chunks.append(chunk.loc[mask])
    
    if not chunks:
        print("Warning: No matching participants found in disease features file")
        return pd.DataFrame({"eid": list(eids)})
    
    disease_df = pd.concat(chunks).drop_duplicates(subset="eid")
    return disease_df


def build_start_dates(df: pd.DataFrame) -> pd.Series:
    yob = pd.to_numeric(df["p34"], errors="coerce")
    mob = pd.to_numeric(df["p52"], errors="coerce").round().clip(1, 12)
    mob = mob.fillna(MONTH_FALLBACK)
    start_year = yob + 60
    start_date = pd.to_datetime(
        {"year": start_year, "month": mob, "day": 1},
        errors="coerce",
    )
    return start_date


def determine_censoring(df: pd.DataFrame) -> pd.Series:
    origin = df["p40020_i0"].astype(str).str.strip().str.upper()
    censoring = origin.map(CENSORING_MAP)
    return censoring.fillna(DEFAULT_CENSOR_DATE)


def compute_survival_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_date"] = build_start_dates(df)
    df["censor_date"] = determine_censoring(df)
    df["event_date"] = pd.to_datetime(df["p40000_i0"], errors="coerce")

    valid_mask = df["start_date"].notna() & df["censor_date"].notna()
    df = df.loc[valid_mask].copy()

    # Events occurring after censor date are treated as censored
    event_before_censor = df["event_date"] <= df["censor_date"]
    df["event_flag"] = event_before_censor.fillna(False).astype(int)

    df["observation_date"] = df["event_date"].where(df["event_flag"] == 1, df["censor_date"])
    df["duration_days"] = (df["observation_date"] - df["start_date"]).dt.days
    df = df.loc[df["duration_days"] >= 0].copy()

    return df


def engineer_features(
    df: pd.DataFrame,
    header: Sequence[str],
    field_map: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    engineered_columns: List[str] = []
    # Medication-derived features
    med_columns = get_field_columns_from_map(field_map, header, "20003")
    med_flags: Dict[str, pd.Series] = {}
    for feature, keywords in MEDICATION_KEYWORDS.items():
        med_flags[feature] = medication_keyword_flag(df, med_columns, keywords)
    for feature, series in med_flags.items():
        df[feature] = series
        engineered_columns.append(feature)
    ed_med_flag = medication_keyword_flag(df, med_columns, ERECTILE_DYSFUNCTION_MEDS)

    # Load pre-computed disease features (diagnosed before age 60)
    disease_df = load_disease_features(df["eid"].tolist())
    disease_feature_cols = [col for col in disease_df.columns if col != "eid"]
    
    if disease_feature_cols:
        # Merge disease features
        df = df.merge(disease_df, on="eid", how="left")
        # Fill missing disease flags with 0 (no diagnosis)
        for col in disease_feature_cols:
            df[col] = df[col].fillna(0).astype(int)
        
        # For erectile_dysfunction, combine with medication flag
        if "erectile_dysfunction" in disease_feature_cols:
            df["erectile_dysfunction"] = ((df["erectile_dysfunction"] == 1) | (ed_med_flag == 1)).astype(int)
        else:
            df["erectile_dysfunction"] = ed_med_flag
        
        engineered_columns.extend(disease_feature_cols)
        print(f"Loaded {len(disease_feature_cols)} pre-computed disease features")
    else:
        # Fallback: just use erectile dysfunction from medication
        df["erectile_dysfunction"] = ed_med_flag
        engineered_columns.append("erectile_dysfunction")

    # Baseline single-value fields
    for feature, field_id in BASELINE_FIELD_MAP.items():
        cols = get_field_columns_from_map(field_map, header, field_id)
        values = first_available_value(df, cols)
        df[feature] = pd.to_numeric(values, errors="coerce")
        engineered_columns.append(feature)

    # BMI
    bmi_cols = get_field_columns_from_map(field_map, header, "21001")
    bmi_series = pd.to_numeric(first_available_value(df, bmi_cols), errors="coerce")
    if bmi_series.isna().all():
        weight = pd.to_numeric(first_available_value(df, get_field_columns_from_map(field_map, header, "21002")), errors="coerce")
        height_cm = pd.to_numeric(first_available_value(df, get_field_columns_from_map(field_map, header, "50")), errors="coerce")
        height_m = height_cm / 100
        height_m = height_m.where(height_m != 0, pd.NA)
        bmi_series = weight / (height_m ** 2)
    df["bmi"] = bmi_series
    engineered_columns.append("bmi")

    # Biomarkers
    for feature, field_id in BIOMARKER_FIELD_MAP.items():
        cols = get_field_columns_from_map(field_map, header, field_id)
        values = first_available_value(df, cols)
        df[feature] = pd.to_numeric(values, errors="coerce")
        engineered_columns.append(feature)

    # Townsend binary
    townsend_numeric = pd.to_numeric(df.get("townsend_index"), errors="coerce")
    townsend_binary = pd.Series(pd.NA, index=df.index, dtype="Int64")
    mask = townsend_numeric.notna()
    townsend_binary.loc[mask] = (townsend_numeric.loc[mask] > TOWNSEND_THRESHOLD).astype(int)
    df["townsend_index"] = townsend_numeric
    df["townsend_binary"] = townsend_binary
    engineered_columns.append("townsend_binary")

    # Living alone flag
    living_cols = get_field_columns_from_map(field_map, header, "709")
    living_values = pd.to_numeric(first_available_value(df, living_cols), errors="coerce")
    living_flag = pd.Series(pd.NA, index=df.index, dtype="Int64")
    live_mask = living_values.notna()
    living_flag.loc[live_mask] = (living_values.loc[live_mask] == LIVING_ALONE_VALUE).astype(int)
    df["living_alone"] = living_flag
    engineered_columns.append("living_alone")

    # Systolic variation
    systolic_cols = get_field_columns_from_map(field_map, header, "4080")
    systolic_cols = [col for col in systolic_cols if col in df.columns]
    if systolic_cols:
        systolic_values = pd.DataFrame(
            {col: pd.to_numeric(df[col], errors="coerce") for col in systolic_cols},
            index=df.index,
        )
        variation = systolic_values.std(axis=1, skipna=True, ddof=0)
    else:
        variation = pd.Series(pd.NA, index=df.index, dtype="float64")
    df["systolic_variation"] = variation
    engineered_columns.append("systolic_variation")

    return df, engineered_columns


MIN_VARIANCE_THRESHOLD = 1e-6  # Features with variance below this are considered constant
MIN_NON_NULL_FRACTION = 0.01  # Features with less than 1% non-null values are dropped


def filter_low_quality_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_variance: float = MIN_VARIANCE_THRESHOLD,
    min_non_null_frac: float = MIN_NON_NULL_FRACTION,
) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    """
    Filter out features with all NaN values or near-zero variance.

    Returns:
        - Filtered DataFrame
        - List of retained feature columns
        - Dict with lists of removed features by reason
    """
    removed: Dict[str, List[str]] = {"all_nan": [], "low_variance": [], "low_coverage": []}
    retained: List[str] = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        series = df[col]
        non_null_count = series.notna().sum()
        non_null_frac = non_null_count / len(series)

        # Check for all NaN
        if non_null_count == 0:
            removed["all_nan"].append(col)
            continue

        # Check for low coverage (too many missing values)
        if non_null_frac < min_non_null_frac:
            removed["low_coverage"].append(col)
            continue

        # Check for near-zero variance (constant or nearly constant)
        variance = series.var(skipna=True)
        if variance is not None and variance < min_variance:
            removed["low_variance"].append(col)
            continue

        retained.append(col)

    # Report removed features
    total_removed = sum(len(v) for v in removed.values())
    if total_removed > 0:
        print(f"Filtered out {total_removed} low-quality features:")
        if removed["all_nan"]:
            print(f"  - All NaN ({len(removed['all_nan'])}): {removed['all_nan']}")
        if removed["low_coverage"]:
            print(f"  - Low coverage <{min_non_null_frac*100:.0f}% ({len(removed['low_coverage'])}): {removed['low_coverage']}")
        if removed["low_variance"]:
            print(f"  - Near-zero variance ({len(removed['low_variance'])}): {removed['low_variance']}")

    # Keep only retained columns plus required columns
    keep_cols = ["eid", "duration_days", "event_flag"] + retained
    df_filtered = df[keep_cols].copy()

    return df_filtered, retained, removed


def load_feature_subset(eids: Sequence[int], feature_cols: Sequence[str], chunksize: int = 5000) -> pd.DataFrame:
    usecols = ["eid"] + list(feature_cols)
    ids_set: Set[int] = set(map(int, eids))
    collected = []
    for chunk in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk["eid"] = chunk["eid"].astype(int)
        mask = chunk["eid"].isin(ids_set)
        if mask.any():
            collected.append(chunk.loc[mask])
    if not collected:
        raise RuntimeError("No feature rows were collected for the sampled participants.")
    features = pd.concat(collected).drop_duplicates(subset="eid")
    return features


def preprocess(sample_size: int = 10_000, random_state: int = 42) -> None:
    header = load_header(DATA_PATH)
    feature_cols, field_map, missing_fields = gather_feature_columns(header)
    required_fields = ENGINEERING_FIELD_IDS
    feature_cols, engineering_missing_fields = extend_feature_columns(header, field_map, feature_cols, required_fields)

    base_df = pd.read_csv(DATA_PATH, usecols=BASE_COLUMNS, low_memory=False)
    survival_df = compute_survival_targets(base_df)

    available = len(survival_df)
    if available == 0:
        raise RuntimeError("No participants with valid survival targets were found.")

    sample_n = min(sample_size, available)
    sampled = survival_df.sample(n=sample_n, random_state=random_state)

    overlap_cols = [col for col in feature_cols if col in sampled.columns]
    sampled = sampled.drop(columns=overlap_cols, errors="ignore")

    features_df = load_feature_subset(sampled["eid"].tolist(), feature_cols)
    merged = sampled.merge(features_df, on="eid", how="left")
    merged, engineered_columns = engineer_features(merged, header, field_map)

    # Only include engineered features in the output (not raw UKB columns like p30760_i0)
    df_sample = merged[["eid", "duration_days", "event_flag"] + engineered_columns].reset_index(drop=True)

    # Filter out low-quality features (all NaN, low coverage, or near-zero variance)
    df_sample, retained_features, removed_features = filter_low_quality_features(
        df_sample, engineered_columns
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(OUTPUT_DATA, index=False)

    metadata = {
        "total_available_rows": available,
        "sample_size": sample_n,
        "missing_comparator_fields": missing_fields,
        "engineering_field_gaps": engineering_missing_fields,
        "excluded_comparator_fields": EXCLUDED_FIELDS,
        "feature_columns": retained_features,
        "removed_features": removed_features,
        "feature_field_map": field_map,
        "base_columns": BASE_COLUMNS,
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote preprocessed dataset to {OUTPUT_DATA} ({sample_n} rows)")
    print(f"Wrote metadata to {OUTPUT_METADATA}")


def main():
    parser = argparse.ArgumentParser(description="Prepare AutoPrognosis survival dataset.")
    parser.add_argument("--sample-size", type=int, default=10_000, help="Number of participants to sample.")
    parser.add_argument("--all", action="store_true",
                        help="Use all participants (overrides --sample-size).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()
    sample_size = 999_999_999 if args.all else args.sample_size
    preprocess(sample_size=sample_size, random_state=args.random_state)


if __name__ == "__main__":
    main()
