"""
Benchmarking (AutoPrognosis) evaluation adapter for the shared cohort.

This script:
  1. Loads the shared cohort split (train/val/test eids).
  2. Loads the survival dataset, restricted to the shared split.
  3. Trains an AutoPrognosis survival model on the train eids.
  4. Evaluates on val/test eids.
  5. Saves metrics for the unified comparison.

Usage:
    python evaluation/evaluate_benchmarking.py [--study-name death_shared]
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarking"))

COHORT_SPLIT = PROJECT_ROOT / "evaluation" / "cohort_split.json"
SURVIVAL_CSV = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "benchmarking_results"


def load_cohort():
    with open(COHORT_SPLIT) as f:
        return json.load(f)


def prepare_dataset(cohort, split="train"):
    """Load the survival dataset filtered to the given split."""
    df = pd.read_csv(SURVIVAL_CSV)
    df["eid"] = df["eid"].astype(int)
    eid_set = set(cohort[f"{split}_eids"])
    return df[df["eid"].isin(eid_set)].copy()


def evaluate_benchmarking(args):
    """Train and evaluate AutoPrognosis on the shared cohort split."""
    cohort = load_cohort()

    df_train = prepare_dataset(cohort, "train")
    df_val = prepare_dataset(cohort, "val")
    df_test = prepare_dataset(cohort, "test")

    target_col = "event_flag"
    time_col = "duration_days"

    # Feature columns: everything except eid, target, time
    exclude = {"eid", target_col, time_col}
    feature_cols = [c for c in df_train.columns if c not in exclude]

    # Keep only numeric features
    numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_cols

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Event rate - Train: {df_train[target_col].mean():.3f}, "
          f"Val: {df_val[target_col].mean():.3f}, "
          f"Test: {df_test[target_col].mean():.3f}")

    # Impute missing values
    train_medians = df_train[feature_cols].median(numeric_only=True)
    for df in [df_train, df_val, df_test]:
        df[feature_cols] = df[feature_cols].fillna(train_medians).fillna(0)

    # Derive time horizons from training data
    durations = df_train[time_col]
    time_horizons = np.quantile(durations, [0.25, 0.5, 0.75]).astype(int).tolist()
    time_horizons = sorted(set(h for h in time_horizons if h > 0))
    print(f"Time horizons: {time_horizons}")

    # Try to use AutoPrognosis if available
    results = {}
    try:
        from survival_baselines import train_and_evaluate_cox
        results = train_and_evaluate_cox(
            df_train, df_test, feature_cols, target_col, time_col, time_horizons
        )
    except ImportError:
        print("AutoPrognosis not available. Using lifelines CoxPH as fallback...")
        results = _evaluate_with_lifelines(
            df_train, df_val, df_test, feature_cols, target_col, time_col, time_horizons
        )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "benchmarking_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return results


def _evaluate_with_lifelines(df_train, df_val, df_test, feature_cols, target_col, time_col, time_horizons):
    """
    Fallback evaluation using lifelines CoxPH when AutoPrognosis is not available.
    Computes C-index and time-dependent AUC.
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        print("lifelines not available either. Install with: pip install lifelines")
        return {"error": "No survival library available"}

    # Prepare training data
    train_data = df_train[feature_cols + [target_col, time_col]].copy()
    train_data = train_data.rename(columns={target_col: "event", time_col: "duration"})

    # Fit Cox PH
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(train_data, duration_col="duration", event_col="event")
    except Exception as e:
        print(f"CoxPH fitting failed: {e}")
        return {"error": str(e)}

    # Evaluate on test set
    results = {"method": "benchmarking_cox_ph", "time_horizons": time_horizons}

    for split_name, df_split in [("val", df_val), ("test", df_test)]:
        split_data = df_split[feature_cols + [target_col, time_col]].copy()

        # Predict risk scores (higher = more risk)
        risk_scores = cph.predict_partial_hazard(split_data[feature_cols])

        # C-index
        c_idx = concordance_index(
            split_data[time_col],
            -risk_scores.values,  # negative because CI expects lower = worse
            split_data[target_col],
        )

        results[f"{split_name}_c_index"] = float(c_idx)
        print(f"  {split_name} C-index: {c_idx:.4f}")

        # Time-dependent AUC at each horizon
        from sksurv.metrics import cumulative_dynamic_auc
        try:
            # Convert to structured array for sksurv
            y_train = np.array(
                [(bool(e), d) for e, d in zip(df_train[target_col], df_train[time_col])],
                dtype=[("event", bool), ("duration", float)],
            )
            y_split = np.array(
                [(bool(e), d) for e, d in zip(df_split[target_col], df_split[time_col])],
                dtype=[("event", bool), ("duration", float)],
            )
            aucs, mean_auc = cumulative_dynamic_auc(
                y_train, y_split, risk_scores.values.flatten(), time_horizons,
            )
            results[f"{split_name}_td_auc"] = {str(h): float(a) for h, a in zip(time_horizons, aucs)}
            results[f"{split_name}_mean_td_auc"] = float(mean_auc)
            print(f"  {split_name} Mean TD-AUC: {mean_auc:.4f}")
        except Exception as e:
            print(f"  Warning: TD-AUC computation failed: {e}")
            results[f"{split_name}_td_auc"] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmarking on shared cohort.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--study-name", type=str, default="death_shared_cohort")
    args = parser.parse_args()

    evaluate_benchmarking(args)


if __name__ == "__main__":
    main()
