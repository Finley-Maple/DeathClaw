"""
Survival model training and evaluation for embedding-based methods (3 & 4).

Takes embeddings (from Qwen text or trajectory token+age) keyed by eid,
merges with baseline features from the survival dataset, trains a survival
model on the train split, and evaluates on val/test.

Usage:
    # Method 3: text embeddings
    python evaluation/evaluate_embedding_survival.py \
        --embedding-dir data/preprocessed/embeddings_text \
        --tag patient \
        --method-name "text_embedding"

    # Method 4: trajectory embeddings
    python evaluation/evaluate_embedding_survival.py \
        --embedding-dir data/preprocessed/embeddings_traj \
        --tag trajectory \
        --method-name "trajectory_embedding"
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

COHORT_SPLIT = PROJECT_ROOT / "evaluation" / "cohort_split.json"
SURVIVAL_CSV = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cohort():
    with open(COHORT_SPLIT) as f:
        return json.load(f)


def load_embeddings(embedding_dir: Path, tag: str) -> dict:
    """Load {eid: np.ndarray} from a .npz file."""
    npz_path = embedding_dir / f"{tag}_embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {npz_path}")
    data = np.load(npz_path)
    return {int(k): data[k] for k in data.files}


def build_feature_matrix(
    eids: list,
    embeddings: dict,
    survival_df: pd.DataFrame,
    baseline_cols: list,
) -> tuple:
    """
    Build feature matrix by concatenating embeddings with baseline features.

    Returns:
        X: np.ndarray of shape (n, emb_dim + n_baseline)
        T: np.ndarray of durations
        E: np.ndarray of event flags
        valid_eids: list of eids that have both embeddings and survival data
    """
    surv_indexed = survival_df.set_index("eid")

    rows_x = []
    rows_t = []
    rows_e = []
    valid_eids = []

    for eid in eids:
        if eid not in embeddings:
            continue
        if eid not in surv_indexed.index:
            continue

        emb = embeddings[eid]
        surv_row = surv_indexed.loc[eid]

        # Baseline features
        baseline = surv_row[baseline_cols].values.astype(float)
        # Replace NaN with 0 in baseline
        baseline = np.nan_to_num(baseline, nan=0.0)

        combined = np.concatenate([emb.flatten(), baseline])
        rows_x.append(combined)
        rows_t.append(float(surv_row["duration_days"]))
        rows_e.append(int(surv_row["event_flag"]))
        valid_eids.append(eid)

    X = np.array(rows_x, dtype=np.float32)
    T = np.array(rows_t, dtype=np.float64)
    E = np.array(rows_e, dtype=np.int32)

    return X, T, E, valid_eids


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(args):
    """Train a survival model on embeddings + baselines; evaluate on val/test."""
    cohort = load_cohort()

    # Load survival data
    surv_df = pd.read_csv(SURVIVAL_CSV)
    surv_df["eid"] = surv_df["eid"].astype(int)

    # Baseline feature columns (non-disease, non-target)
    target_cols = {"eid", "event_flag", "duration_days"}
    baseline_cols = [c for c in surv_df.columns
                     if c not in target_cols
                     and surv_df[c].dtype in [np.float64, np.int64, float, int]]

    # Load embeddings
    print(f"Loading embeddings from {args.embedding_dir} (tag={args.tag})...")
    embeddings = load_embeddings(Path(args.embedding_dir), args.tag)
    print(f"  Loaded {len(embeddings)} embeddings")

    sample_emb = next(iter(embeddings.values()))
    emb_dim = sample_emb.shape[0] if sample_emb.ndim > 0 else 0
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Baseline features: {len(baseline_cols)}")

    # Build matrices for each split
    X_train, T_train, E_train, eids_train = build_feature_matrix(
        cohort["train_eids"], embeddings, surv_df, baseline_cols
    )
    X_val, T_val, E_val, eids_val = build_feature_matrix(
        cohort["val_eids"], embeddings, surv_df, baseline_cols
    )
    X_test, T_test, E_test, eids_test = build_feature_matrix(
        cohort["test_eids"], embeddings, surv_df, baseline_cols
    )

    print(f"\nSplit sizes (with valid embeddings):")
    print(f"  Train: {len(eids_train)} / {len(cohort['train_eids'])}")
    print(f"  Val:   {len(eids_val)} / {len(cohort['val_eids'])}")
    print(f"  Test:  {len(eids_test)} / {len(cohort['test_eids'])}")
    print(f"  Event rates: train={E_train.mean():.3f}, val={E_val.mean():.3f}, test={E_test.mean():.3f}")

    # Standardize features (using train stats)
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std < 1e-8] = 1.0  # avoid div by zero

    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # Time horizons
    time_horizons = np.quantile(T_train[T_train > 0], [0.25, 0.5, 0.75]).astype(int).tolist()
    time_horizons = sorted(set(h for h in time_horizons if h > 0))
    print(f"  Time horizons: {time_horizons}")

    # Train CoxPH using lifelines
    results = _train_cox_ph(
        X_train, T_train, E_train,
        X_val, T_val, E_val,
        X_test, T_test, E_test,
        time_horizons,
        method_name=args.method_name,
    )

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{args.method_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return results


def _train_cox_ph(X_train, T_train, E_train, X_val, T_val, E_val,
                  X_test, T_test, E_test, time_horizons, method_name="embedding"):
    """Train CoxPH and compute metrics."""
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        print("lifelines not installed. Install: pip install lifelines")
        return {"error": "lifelines not available"}

    # Create DataFrame for lifelines
    n_features = X_train.shape[1]
    col_names = [f"f{i}" for i in range(n_features)]

    df_train = pd.DataFrame(X_train, columns=col_names)
    df_train["duration"] = T_train
    df_train["event"] = E_train

    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)  # Elastic net to handle high dim
    try:
        cph.fit(df_train, duration_col="duration", event_col="event", show_progress=True)
    except Exception as e:
        print(f"CoxPH fitting failed: {e}")
        # Try with higher penalizer
        cph = CoxPHFitter(penalizer=1.0, l1_ratio=0.9)
        try:
            cph.fit(df_train, duration_col="duration", event_col="event", show_progress=True)
        except Exception as e2:
            return {"error": str(e2)}

    results = {"method": method_name, "time_horizons": time_horizons}

    for split_name, X_split, T_split, E_split in [
        ("val", X_val, T_val, E_val),
        ("test", X_test, T_test, E_test),
    ]:
        df_split = pd.DataFrame(X_split, columns=col_names)

        # Predict risk
        risk_scores = cph.predict_partial_hazard(df_split)

        # C-index
        c_idx = concordance_index(T_split, -risk_scores.values.flatten(), E_split)
        results[f"{split_name}_c_index"] = float(c_idx)
        print(f"  {split_name} C-index: {c_idx:.4f}")

        # Time-dependent AUC
        try:
            from sksurv.metrics import cumulative_dynamic_auc

            y_train = np.array(
                [(bool(e), d) for e, d in zip(E_train, T_train)],
                dtype=[("event", bool), ("duration", float)],
            )
            y_split = np.array(
                [(bool(e), d) for e, d in zip(E_split, T_split)],
                dtype=[("event", bool), ("duration", float)],
            )

            # Filter valid horizons
            valid_horizons = [h for h in time_horizons if h < T_split.max()]
            if valid_horizons:
                aucs, mean_auc = cumulative_dynamic_auc(
                    y_train, y_split, risk_scores.values.flatten(), valid_horizons,
                )
                results[f"{split_name}_td_auc"] = {
                    str(h): float(a) for h, a in zip(valid_horizons, aucs)
                }
                results[f"{split_name}_mean_td_auc"] = float(mean_auc)
                print(f"  {split_name} Mean TD-AUC: {mean_auc:.4f}")
            else:
                results[f"{split_name}_td_auc"] = None
        except ImportError:
            print("  sksurv not available for TD-AUC")
            results[f"{split_name}_td_auc"] = None

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate survival model on embedding + baseline features."
    )
    parser.add_argument("--embedding-dir", type=str, required=True,
                        help="Directory containing {tag}_embeddings.npz")
    parser.add_argument("--tag", type=str, default="patient",
                        help="Embedding file prefix (e.g. 'patient', 'trajectory')")
    parser.add_argument("--method-name", type=str, default="embedding",
                        help="Name for this method in results")
    parser.add_argument("--output-dir", type=str,
                        default=str(DEFAULT_OUTPUT / "embedding_results"))
    args = parser.parse_args()

    train_and_evaluate(args)


if __name__ == "__main__":
    main()
