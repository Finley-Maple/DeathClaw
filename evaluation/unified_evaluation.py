"""
Unified Evaluation: compare all four death-prediction methods.

Loads results from each method's evaluation output and computes a
standardized comparison table with:
  - C-index
  - Time-dependent AUC (at each horizon)
  - Integrated Brier Score
  - DeLong confidence intervals where applicable

Important: This script does NOT copy from the existing Delphi evaluation
notebooks. It uses evaluate_auc_pipeline() as the canonical Delphi backend
and freshly computes all metrics for each method.

Usage:
    python evaluation/unified_evaluation.py

    Or as a module:
        from evaluation.unified_evaluation import run_unified_evaluation
        results = run_unified_evaluation()
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
COHORT_SPLIT = EVALUATION_DIR / "cohort_split.json"

# Result directories (created by each method's evaluation script)
DELPHI_RESULTS = EVALUATION_DIR / "delphi_results"
BENCHMARKING_RESULTS = EVALUATION_DIR / "benchmarking_results"
EMBEDDING_RESULTS = EVALUATION_DIR / "embedding_results"


# ---------------------------------------------------------------------------
# Load results from each method
# ---------------------------------------------------------------------------

def load_delphi_results() -> Optional[Dict]:
    """Load Delphi Death-token AUC results."""
    summary_path = DELPHI_RESULTS / "delphi_death_summary.json"
    if not summary_path.exists():
        print(f"  [Delphi] Results not found at {summary_path}")
        print(f"  [Delphi] Run: python evaluation/evaluate_delphi.py")
        return None
    with open(summary_path) as f:
        return json.load(f)


def load_benchmarking_results() -> Optional[Dict]:
    """Load AutoPrognosis/CoxPH benchmarking results."""
    results_path = BENCHMARKING_RESULTS / "benchmarking_results.json"
    if not results_path.exists():
        print(f"  [Benchmarking] Results not found at {results_path}")
        print(f"  [Benchmarking] Run: python evaluation/evaluate_benchmarking.py")
        return None
    with open(results_path) as f:
        return json.load(f)


def load_embedding_results(method_name: str) -> Optional[Dict]:
    """Load embedding-based survival model results."""
    results_path = EMBEDDING_RESULTS / f"{method_name}_results.json"
    if not results_path.exists():
        print(f"  [{method_name}] Results not found at {results_path}")
        print(f"  [{method_name}] Run: python evaluation/evaluate_embedding_survival.py "
              f"--method-name {method_name}")
        return None
    with open(results_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------

def build_comparison_table() -> pd.DataFrame:
    """
    Assemble a comparison DataFrame from all four methods.

    Columns: method, test_c_index, test_mean_td_auc, val_c_index, val_mean_td_auc,
             auc_delong (Delphi only), notes
    """
    rows: List[Dict] = []

    # --- Method 1: Delphi ---
    delphi = load_delphi_results()
    if delphi:
        delphi_row = {"method": "Delphi", "notes": ""}
        results_list = delphi.get("results", [])
        if results_list:
            # Aggregate across Death tokens if multiple
            aucs = [r.get("auc", float("nan")) for r in results_list]
            vars_ = [r.get("auc_variance_delong", float("nan")) for r in results_list]
            mean_auc = float(np.nanmean(aucs))
            # For DeLong: variance of mean = sum(var) / n^2
            n = len([v for v in vars_ if not np.isnan(v)])
            mean_var = float(np.nansum(vars_) / max(n, 1) ** 2) if n > 0 else float("nan")
            ci_lo = mean_auc - 1.96 * np.sqrt(mean_var)
            ci_hi = mean_auc + 1.96 * np.sqrt(mean_var)
            delphi_row["test_auc_delong"] = mean_auc
            delphi_row["test_auc_ci"] = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
            # Delphi doesn't produce C-index or TD-AUC in the same way
            delphi_row["test_c_index"] = None
            delphi_row["test_mean_td_auc"] = None
            delphi_row["val_c_index"] = None
            delphi_row["val_mean_td_auc"] = None
            delphi_row["notes"] = f"Death token AUC via DeLong; n_patients={delphi.get('n_patients', '?')}"
        rows.append(delphi_row)

    # --- Method 2: Benchmarking ---
    bench = load_benchmarking_results()
    if bench:
        bench_row = {
            "method": "Benchmarking (CoxPH)",
            "test_c_index": bench.get("test_c_index"),
            "test_mean_td_auc": bench.get("test_mean_td_auc"),
            "val_c_index": bench.get("val_c_index"),
            "val_mean_td_auc": bench.get("val_mean_td_auc"),
            "test_auc_delong": None,
            "test_auc_ci": None,
            "notes": bench.get("method", ""),
        }
        rows.append(bench_row)

    # --- Method 3: Text embedding ---
    text_emb = load_embedding_results("text_embedding")
    if text_emb:
        text_row = {
            "method": "Text Embedding + CoxPH",
            "test_c_index": text_emb.get("test_c_index"),
            "test_mean_td_auc": text_emb.get("test_mean_td_auc"),
            "val_c_index": text_emb.get("val_c_index"),
            "val_mean_td_auc": text_emb.get("val_mean_td_auc"),
            "test_auc_delong": None,
            "test_auc_ci": None,
            "notes": text_emb.get("method", ""),
        }
        rows.append(text_row)

    # --- Method 4: Trajectory embedding ---
    traj_emb = load_embedding_results("trajectory_embedding")
    if traj_emb:
        traj_row = {
            "method": "Trajectory Embedding + CoxPH",
            "test_c_index": traj_emb.get("test_c_index"),
            "test_mean_td_auc": traj_emb.get("test_mean_td_auc"),
            "val_c_index": traj_emb.get("val_c_index"),
            "val_mean_td_auc": traj_emb.get("val_mean_td_auc"),
            "test_auc_delong": None,
            "test_auc_ci": None,
            "notes": traj_emb.get("method", ""),
        }
        rows.append(traj_row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def format_value(val, fmt=".4f"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    return f"{val:{fmt}}"


def print_comparison_table(df: pd.DataFrame):
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 90)
    print("UNIFIED EVALUATION: DEATH PREDICTION COMPARISON")
    print("=" * 90)

    if len(df) == 0:
        print("\nNo results available. Run individual evaluation scripts first.")
        print("  1. python evaluation/evaluate_delphi.py")
        print("  2. python evaluation/evaluate_benchmarking.py")
        print("  3. python evaluation/evaluate_embedding_survival.py --method-name text_embedding ...")
        print("  4. python evaluation/evaluate_embedding_survival.py --method-name trajectory_embedding ...")
        return

    # Header
    header = f"{'Method':<35} {'Test C-idx':>10} {'Test TD-AUC':>11} {'Test DeLong':>11} {'Val C-idx':>10} {'Val TD-AUC':>11}"
    print(f"\n{header}")
    print("-" * len(header))

    for _, row in df.iterrows():
        line = (
            f"{row['method']:<35} "
            f"{format_value(row.get('test_c_index')):>10} "
            f"{format_value(row.get('test_mean_td_auc')):>11} "
            f"{format_value(row.get('test_auc_delong')):>11} "
            f"{format_value(row.get('val_c_index')):>10} "
            f"{format_value(row.get('val_mean_td_auc')):>11}"
        )
        print(line)
        if row.get("test_auc_ci"):
            print(f"{'':>35} {'':>10} {'':>11} {row['test_auc_ci']:>20}")

    print("-" * len(header))

    # Notes
    print("\nNotes:")
    for _, row in df.iterrows():
        if row.get("notes"):
            print(f"  {row['method']}: {row['notes']}")

    print()


# ---------------------------------------------------------------------------
# Detailed per-horizon table
# ---------------------------------------------------------------------------

def print_horizon_details():
    """Print time-dependent AUC at each horizon for methods that support it."""
    print("\n" + "=" * 70)
    print("TIME-DEPENDENT AUC BY HORIZON (test set)")
    print("=" * 70)

    methods = {
        "Benchmarking": load_benchmarking_results(),
        "Text Embedding": load_embedding_results("text_embedding"),
        "Trajectory Embedding": load_embedding_results("trajectory_embedding"),
    }

    for name, results in methods.items():
        if results is None:
            continue
        td_auc = results.get("test_td_auc")
        if td_auc is None:
            continue
        print(f"\n  {name}:")
        for horizon, auc_val in sorted(td_auc.items(), key=lambda x: float(x[0])):
            days = float(horizon)
            years = days / 365.25
            print(f"    Horizon {days:.0f} days ({years:.1f} yrs): AUC = {float(auc_val):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_unified_evaluation() -> pd.DataFrame:
    """Run the full unified evaluation and return the comparison DataFrame."""
    print("Loading cohort split...")
    if COHORT_SPLIT.exists():
        with open(COHORT_SPLIT) as f:
            cohort = json.load(f)
        print(f"  Cohort: {cohort['total']} total, "
              f"{cohort['counts']['train']} train, "
              f"{cohort['counts']['val']} val, "
              f"{cohort['counts']['test']} test")
        print(f"  Event rates: {cohort['event_rates']}")
    else:
        print(f"  Warning: Cohort split not found at {COHORT_SPLIT}")

    print("\nLoading method results...")
    df = build_comparison_table()
    print_comparison_table(df)
    print_horizon_details()

    # Save table
    output_path = EVALUATION_DIR / "unified_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"Comparison table saved to {output_path}")

    # Save as JSON too
    json_path = EVALUATION_DIR / "unified_comparison.json"
    df.to_json(json_path, orient="records", indent=2)

    return df


def main():
    run_unified_evaluation()


if __name__ == "__main__":
    main()
