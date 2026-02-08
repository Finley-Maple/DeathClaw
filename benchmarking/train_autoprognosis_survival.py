import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Parallelization: increase for faster search on multi-core machines
# Set to number of CPU cores available (e.g. 4, 8, or -1 for all cores)
os.environ.setdefault("N_OPT_JOBS", "4")
os.environ.setdefault("N_LEARNER_JOBS", "4")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "1")

from autoprognosis.studies.risk_estimation import RiskEstimationStudy
from autoprognosis.utils import redis as redis_utils
from autoprognosis.explorers.core.optimizers import bayesian as bayesian_optimizer


# Redis backend configuration
# Set USE_REDIS=True if you have Redis running locally (redis-server)
USE_REDIS = True

if not USE_REDIS:
    # Disable Redis when not available (e.g., in sandbox environments)
    class _NoRedisBackend:
        def __init__(self, *args, **kwargs):
            self._optuna_storage = None

        def optuna(self):
            return None

        def client(self):
            raise RuntimeError("Redis backend disabled. Set USE_REDIS=true to enable.")

    redis_utils.RedisBackend = _NoRedisBackend
    bayesian_optimizer.RedisBackend = _NoRedisBackend
else:
    print("[AutoPrognosis] Redis backend ENABLED for distributed HP optimization.", flush=True)


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


def derive_time_horizons(durations: pd.Series, quantiles: tuple[float, ...]) -> list[int]:
    horizons = np.quantile(durations, quantiles).astype(int).tolist()
    horizons = sorted(set(int(h) for h in horizons if h > 0))
    if len(horizons) == 0:
        raise ValueError("Failed to derive positive time horizons from durations.")
    return horizons


def train_survival_model(args: argparse.Namespace) -> None:
    log_message("train_survival_model invoked")
    print("[AutoPrognosis] train_survival_model starting...", flush=True)
    df = load_dataset()
    log_message("dataset_loaded")
    target_col = "event_flag"
    time_col = "duration_days"
    feature_df = df.drop(columns=[target_col, time_col])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    dropped_cols = [c for c in feature_df.columns if c not in numeric_cols]
    if dropped_cols:
        log_message(f"dropping_non_numeric_cols={len(dropped_cols)}")
    if args.max_features is not None and len(numeric_cols) > args.max_features:
        numeric_cols = numeric_cols[: args.max_features]
        log_message(f"feature_cap={args.max_features}")

    feature_df = feature_df[numeric_cols].copy()
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    feature_df = feature_df.fillna(0)
    feature_cols = feature_df.columns.tolist()
    df = pd.concat([feature_df, df[[target_col, time_col]].reset_index(drop=True)], axis=1)

    if args.train_sample_size is not None and args.train_sample_size < len(df):
        df = df.sample(n=args.train_sample_size, random_state=args.random_state).reset_index(drop=True)
        log_message(f"sampled_rows={len(df)}")

    time_horizons = derive_time_horizons(
        df[time_col], quantiles=args.quantiles
    )
    log_message(f"horizons={time_horizons}")

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    print(f"[AutoPrognosis] Using {len(df)} rows, {len(feature_cols)} features.", flush=True)
    print(f"[AutoPrognosis] Time horizons: {time_horizons}", flush=True)

    log_message("init_study_start")
    study = RiskEstimationStudy(
        study_name=args.study_name,
        dataset=df[[*feature_cols, target_col, time_col]],
        target=target_col,
        time_to_event=time_col,
        time_horizons=time_horizons,
        num_iter=args.num_iter,
        num_study_iter=args.num_study_iter,
        num_ensemble_iter=args.num_ensemble_iter,
        timeout=args.timeout,
        workspace=WORKSPACE,
        random_state=args.random_state,
        sample_for_search=True,
        max_search_sample_size=args.max_search_sample_size,
        risk_estimators=args.risk_estimators,
        imputers=args.imputers,
        score_threshold=args.score_threshold,
    )
    log_message("init_study_complete")
    print("[AutoPrognosis] Study initialized, starting fit()...", flush=True)

    model = study.fit()
    log_message("study_fit_complete")
    print("[AutoPrognosis] Study fit complete.", flush=True)
    if model is None:
        raise RuntimeError(
            "AutoPrognosis could not find a model above the requested score threshold. "
            "Try lowering --score-threshold or increasing --num-iter."
        )

    # Save the model
    model_path = WORKSPACE / f"{args.study_name}_model.pkl"
    model.save(model_path)
    log_message(f"model_saved={model_path}")
    print(f"[AutoPrognosis] Model saved to {model_path}", flush=True)

    # Save metadata (feature columns, time horizons, etc.) for evaluation
    metadata = {
        "feature_cols": feature_cols,
        "time_horizons": time_horizons,
        "target_col": target_col,
        "time_col": time_col,
        "study_name": args.study_name,
        "random_state": args.random_state,
    }
    metadata_path = WORKSPACE / f"{args.study_name}_metadata.json"
    with metadata_path.open("w") as fh:
        json.dump(metadata, fh, indent=2)
    log_message(f"metadata_saved={metadata_path}")
    print(f"[AutoPrognosis] Metadata saved to {metadata_path}", flush=True)

    print("[AutoPrognosis] Training complete!", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AutoPrognosis survival model on the prepared dataset.")
    parser.add_argument("--study-name", type=str, default="ukb_death_registry_survival_10k")
    parser.add_argument("--num-iter", type=int, default=20, help="Max optimizer trials per estimator (more = better tuning).")
    parser.add_argument("--num-study-iter", type=int, default=3, help="Outer study iterations.")
    parser.add_argument("--num-ensemble-iter", type=int, default=10, help="Ensemble search iterations.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-estimator search timeout in seconds (increase for full dataset).")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-search-sample-size", type=int, default=10000,
                        help="Subsample used during hyperparameter search (larger = slower but better tuning).")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=None,
        help="Subsample size for training. Use None or omit to train on full dataset.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of numeric features. Use None to keep all features.",
    )
    parser.add_argument(
        "--risk-estimators",
        nargs="+",
        default=["cox_ph", "coxnet", "weibull_aft"],
        help="Subset of survival models to search over.",
    )
    parser.add_argument(
        "--imputers",
        nargs="*",
        default=[],
        help="Imputation plugins to consider (leave empty after external imputation).",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=(0.25, 0.5, 0.75),
        help="Quantiles of duration to use as evaluation horizons.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_survival_model(args)
