"""
Define a shared cohort (eid list) and train/val/test split for all four methods.

The cohort is derived from the existing autoprognosis_survival_dataset.csv, which
already contains participants with valid survival targets (death after 60).

Usage:
    python evaluation/cohort_split.py [--random-state 42] [--train-frac 0.70]
                                       [--val-frac 0.15] [--test-frac 0.15]
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURVIVAL_DATASET = PROJECT_ROOT / "benchmarking" / "autoprognosis_survival_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "cohort_split.json"


def create_cohort_split(
    survival_csv: Path = SURVIVAL_DATASET,
    output_path: Path = OUTPUT_PATH,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Read the survival dataset, extract all eids, and split into
    train / val / test sets with the given fractions.

    Returns and saves a JSON dict:
        {
            "random_state": int,
            "total": int,
            "fractions": {"train": ..., "val": ..., "test": ...},
            "train_eids": [...],
            "val_eids": [...],
            "test_eids": [...],
            "event_rate": float,  # overall death rate for reference
        }
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"

    if not survival_csv.exists():
        print(f"Error: Survival dataset not found at {survival_csv}")
        print("Run benchmarking/preprocess_survival.py first.")
        sys.exit(1)

    df = pd.read_csv(survival_csv, usecols=["eid", "event_flag", "duration_days"])
    eids = df["eid"].astype(int).values
    event_flags = df["event_flag"].values

    n = len(eids)
    print(f"Loaded {n} participants from {survival_csv.name}")
    print(f"  Death events: {event_flags.sum()} ({event_flags.mean() * 100:.1f}%)")

    # Stratified split: maintain similar event rates across splits
    rng = np.random.RandomState(random_state)
    indices = np.arange(n)

    # Separate event and non-event indices for stratification
    event_idx = indices[event_flags == 1]
    nonevent_idx = indices[event_flags == 0]

    rng.shuffle(event_idx)
    rng.shuffle(nonevent_idx)

    def split_indices(idx_array, train_f, val_f):
        n_arr = len(idx_array)
        n_train = int(round(n_arr * train_f))
        n_val = int(round(n_arr * val_f))
        return (
            idx_array[:n_train],
            idx_array[n_train:n_train + n_val],
            idx_array[n_train + n_val:],
        )

    e_train, e_val, e_test = split_indices(event_idx, train_frac, val_frac)
    ne_train, ne_val, ne_test = split_indices(nonevent_idx, train_frac, val_frac)

    train_idx = np.concatenate([e_train, ne_train])
    val_idx = np.concatenate([e_val, ne_val])
    test_idx = np.concatenate([e_test, ne_test])

    # Shuffle within each split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_eids = eids[train_idx].tolist()
    val_eids = eids[val_idx].tolist()
    test_eids = eids[test_idx].tolist()

    # Verify no overlap
    assert len(set(train_eids) & set(val_eids)) == 0, "Train/val overlap!"
    assert len(set(train_eids) & set(test_eids)) == 0, "Train/test overlap!"
    assert len(set(val_eids) & set(test_eids)) == 0, "Val/test overlap!"
    assert len(train_eids) + len(val_eids) + len(test_eids) == n, "Eids lost!"

    # Compute per-split event rates for verification
    train_er = event_flags[train_idx].mean()
    val_er = event_flags[val_idx].mean()
    test_er = event_flags[test_idx].mean()

    result = {
        "random_state": random_state,
        "total": n,
        "fractions": {
            "train": train_frac,
            "val": val_frac,
            "test": test_frac,
        },
        "counts": {
            "train": len(train_eids),
            "val": len(val_eids),
            "test": len(test_eids),
        },
        "event_rates": {
            "overall": float(event_flags.mean()),
            "train": float(train_er),
            "val": float(val_er),
            "test": float(test_er),
        },
        "train_eids": train_eids,
        "val_eids": val_eids,
        "test_eids": test_eids,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nCohort split saved to {output_path}")
    print(f"  Train: {len(train_eids)} (event rate {train_er:.3f})")
    print(f"  Val:   {len(val_eids)} (event rate {val_er:.3f})")
    print(f"  Test:  {len(test_eids)} (event rate {test_er:.3f})")

    return result


def load_cohort_split(path: Path = OUTPUT_PATH) -> dict:
    """Load a previously saved cohort split."""
    if not path.exists():
        raise FileNotFoundError(
            f"Cohort split not found at {path}. Run evaluation/cohort_split.py first."
        )
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Create shared cohort train/val/test split.")
    parser.add_argument("--survival-csv", type=Path, default=SURVIVAL_DATASET,
                        help="Path to the autoprognosis_survival_dataset.csv file.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Path to save the cohort split JSON.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Train fraction.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction.")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test fraction.")
    args = parser.parse_args()

    create_cohort_split(
        survival_csv=args.survival_csv,
        output_path=args.output,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
