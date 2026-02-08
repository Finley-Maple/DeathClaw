import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Parallelization settings
os.environ.setdefault("N_OPT_JOBS", "4")
os.environ.setdefault("N_LEARNER_JOBS", "4")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "1")

from autoprognosis.utils.tester import evaluate_survival_estimator
from autoprognosis.plugins.prediction.risk_estimation import RiskEstimation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
WORKSPACE = PROJECT_ROOT / "benchmarking" / "autoprognosis_workspace"
RUN_LOG = WORKSPACE / "run_log.txt"


def log_message(message: str) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as fh:
        fh.write(f"{message}\n")


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {DATA_PATH}. Run preprocess_survival.py first."
        )
    df = pd.read_csv(DATA_PATH)
    return df


def load_model(model_path: Path):
    """Load a saved AutoPrognosis survival model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_autoprognosis_survival.py first.")
    return RiskEstimation.load(model_path)


def load_metadata(metadata_path: Path) -> dict:
    """Load saved metadata (feature columns, time horizons, etc.)."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run train_autoprognosis_survival.py first.")
    with metadata_path.open("r") as fh:
        return json.load(fh)


def evaluate_survival_model(args: argparse.Namespace) -> None:
    log_message("evaluate_survival_model invoked")
    print("[AutoPrognosis] evaluate_survival_model starting...", flush=True)

    # Load metadata
    metadata_path = WORKSPACE / f"{args.study_name}_metadata.json"
    metadata = load_metadata(metadata_path)
    feature_cols = metadata["feature_cols"]
    time_horizons = metadata["time_horizons"]
    target_col = metadata["target_col"]
    time_col = metadata["time_col"]
    random_state = metadata.get("random_state", args.random_state)

    print(f"[AutoPrognosis] Loaded metadata: {len(feature_cols)} features, horizons={time_horizons}", flush=True)

    # Load model
    model_path = WORKSPACE / f"{args.study_name}_model.pkl"
    model = load_model(model_path)
    log_message("model_loaded")
    print(f"[AutoPrognosis] Model loaded from {model_path}", flush=True)

    # Load dataset
    df = load_dataset()
    log_message("dataset_loaded")

    # Prepare features (same preprocessing as training)
    feature_df = df.drop(columns=[target_col, time_col], errors="ignore")
    
    # Ensure we only use features that were used during training
    missing_cols = [c for c in feature_cols if c not in feature_df.columns]
    if missing_cols:
        raise ValueError(f"Missing {len(missing_cols)} features in dataset: {missing_cols[:5]}...")
    
    feature_df = feature_df[feature_cols].copy()
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    feature_df = feature_df.fillna(0)

    # Optional: subsample for evaluation
    if args.eval_sample_size is not None and args.eval_sample_size < len(df):
        indices = df.sample(n=args.eval_sample_size, random_state=random_state).index
        feature_df = feature_df.loc[indices].reset_index(drop=True)
        df = df.loc[indices].reset_index(drop=True)
        log_message(f"eval_sampled_rows={len(df)}")
        print(f"[AutoPrognosis] Evaluating on {len(df)} samples", flush=True)

    print(f"[AutoPrognosis] Running {args.cv_folds}-fold cross-validation...", flush=True)
    
    cv_results = evaluate_survival_estimator(
        model,
        X=feature_df,
        T=df[time_col],
        Y=df[target_col],
        time_horizons=time_horizons,
        n_folds=args.cv_folds,
        seed=random_state,
    )
    log_message("evaluation_complete")

    print("\n" + "=" * 60)
    print("Evaluation metrics (aggregated):")
    print("=" * 60)
    for metric, summary in cv_results["str"].items():
        print(f"  {metric}: {summary}")

    print("\n" + "=" * 60)
    print("Per-horizon metrics:")
    print("=" * 60)
    for metric, horizons in cv_results["horizons"].items():
        print(f"{metric}:")
        for horizon, values in horizons.items():
            if isinstance(values, (list, tuple)) and len(values) == 2:
                mean, std = values
                print(f"  Horizon {horizon:.1f}: mean={mean:.4f}, std={std:.4f}")
            else:
                print(f"  Horizon {horizon:.1f}: {values}")

    # Optionally save results
    if args.output_path:
        output_path = Path(args.output_path)
        results_to_save = {
            "study_name": args.study_name,
            "cv_folds": args.cv_folds,
            "time_horizons": time_horizons,
            "aggregated_metrics": cv_results["str"],
            "horizon_metrics": {
                metric: {
                    str(h): list(v) if isinstance(v, (list, tuple)) else v
                    for h, v in horizons.items()
                }
                for metric, horizons in cv_results["horizons"].items()
            },
        }
        with output_path.open("w") as fh:
            json.dump(results_to_save, fh, indent=2)
        print(f"\n[AutoPrognosis] Results saved to {output_path}", flush=True)

    print("\n[AutoPrognosis] Evaluation complete!", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained AutoPrognosis survival model.")
    parser.add_argument("--study-name", type=str, default="ukb_death_registry_survival_10k",
                        help="Name of the study (used to locate saved model and metadata).")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of cross-validation folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state (overridden by saved metadata if available).")
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=None,
        help="Subsample size for evaluation. Use None or omit to evaluate on full dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON. If not specified, results are only printed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_survival_model(args)
