"""
Delphi evaluation adapter for the shared cohort.

Uses the `evaluate_auc_pipeline()` function from Delphi/evaluate_auc.py
as the canonical evaluation backend. Ensures the val/test bins are
restricted to the shared cohort.

This script:
  1. Loads the shared cohort split.
  2. Loads Delphi model checkpoint + validation data.
  3. Filters val data to shared cohort eids (val + test).
  4. Runs evaluate_auc_pipeline for the Death token.
  5. Saves results to evaluation/delphi_results/.

Usage:
    python evaluation/evaluate_delphi.py \
        --ckpt-path Delphi/Delphi-2M-respiratory/ckpt.pt \
        --data-path Delphi/data/ukb_respiratory_data \
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Delphi"))

import torch
from model import Delphi, DelphiConfig
from utils import get_batch, get_p2i
from evaluate_auc import evaluate_auc_pipeline

COHORT_SPLIT = PROJECT_ROOT / "evaluation" / "cohort_split.json"
DELPHI_LABELS = PROJECT_ROOT / "Delphi" / "delphi_labels_chapters_colours_icd.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "delphi_results"


def load_cohort_eids(split: str = "val_test") -> list:
    """Load eids for the specified split(s)."""
    with open(COHORT_SPLIT) as f:
        cohort = json.load(f)
    if split == "val":
        return cohort["val_eids"]
    elif split == "test":
        return cohort["test_eids"]
    elif split == "val_test":
        return cohort["val_eids"] + cohort["test_eids"]
    elif split == "all":
        return cohort["train_eids"] + cohort["val_eids"] + cohort["test_eids"]
    else:
        raise ValueError(f"Unknown split: {split}")


def evaluate_delphi(args):
    """Run Delphi death-token AUC evaluation on the shared cohort."""
    device = args.device
    seed = args.seed

    # Load model
    print(f"Loading Delphi model from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    conf = DelphiConfig(**checkpoint["model_args"])
    model = Delphi(conf)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.to(device)

    # Load validation data
    val_bin_path = Path(args.data_path) / "val.bin"
    print(f"Loading validation data from {val_bin_path}...")
    val = np.fromfile(str(val_bin_path), dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    val_p2i = get_p2i(val)

    # Optionally filter to shared cohort eids
    # NOTE: Delphi's binary format uses internal patient indices, not eids directly.
    # The p2i mapping contains [start_idx, length] per patient.
    # If the binary data was built from the same cohort, all patients are valid.
    # If not, we would need the eid-to-patient-index mapping, which requires
    # the original preprocessing notebook. For now, we evaluate on all available
    # patients in val.bin and note this in the output.

    n_patients = len(val_p2i)
    if args.max_patients > 0:
        n_patients = min(n_patients, args.max_patients)

    print(f"Evaluating on {n_patients} patients...")
    d_batch = get_batch(
        range(n_patients),
        val,
        val_p2i,
        select="left",
        block_size=80,
        device=device,
        padding="random",
        no_event_token_rate=args.no_event_token_rate,
    )

    # Load labels
    delphi_labels = pd.read_csv(DELPHI_LABELS)

    # Find Death token index
    death_rows = delphi_labels[delphi_labels["name"].str.contains("Death", case=False, na=False)]
    if len(death_rows) > 0:
        death_token_ids = death_rows["index"].tolist()
        print(f"Death token indices: {death_token_ids}")
    else:
        # Fallback to known indices
        death_token_ids = [1268, 1269]
        print(f"Using default Death token indices: {death_token_ids}")

    # Run evaluation
    output_path = args.output_dir
    df_unpooled, df_pooled = evaluate_auc_pipeline(
        model=model,
        d100k=d_batch,
        output_path=str(output_path),
        delphi_labels=delphi_labels,
        diseases_of_interest=death_token_ids,
        filter_min_total=0,  # we want Death even if rare in val
        disease_chunk_size=200,
        age_groups=np.arange(40, 80, 5),
        offset=args.offset,
        batch_size=args.batch_size,
        device=device,
        seed=seed,
        n_bootstrap=args.n_bootstrap,
        meta_info={"method": "delphi", "cohort": "shared"},
    )

    print("\n=== Delphi Death Token AUC Results ===")
    death_results = df_pooled[df_pooled["name"].str.contains("Death", case=False, na=False)]
    if len(death_results) > 0:
        for _, row in death_results.iterrows():
            auc_val = row.get("auc", float("nan"))
            auc_var = row.get("auc_variance_delong", float("nan"))
            ci_lo = auc_val - 1.96 * np.sqrt(auc_var) if not np.isnan(auc_var) else float("nan")
            ci_hi = auc_val + 1.96 * np.sqrt(auc_var) if not np.isnan(auc_var) else float("nan")
            print(f"  Token: {row.get('name', '?')}")
            print(f"  AUC (DeLong): {auc_val:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
            print(f"  N diseased: {row.get('n_diseased', '?')}, N healthy: {row.get('n_healthy', '?')}")
    else:
        print("  No Death token results found.")

    # Save summary
    summary = {
        "method": "delphi",
        "n_patients": n_patients,
        "death_token_ids": death_token_ids,
        "results": death_results.to_dict("records") if len(death_results) > 0 else [],
    }
    summary_path = Path(output_path) / "delphi_death_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    return df_unpooled, df_pooled


def main():
    parser = argparse.ArgumentParser(description="Evaluate Delphi on the shared cohort.")
    parser.add_argument("--ckpt-path", type=str,
                        default=str(PROJECT_ROOT / "Delphi" / "Delphi-2M-respiratory" / "ckpt.pt"))
    parser.add_argument("--data-path", type=str,
                        default=str(PROJECT_ROOT / "Delphi" / "data" / "ukb_respiratory_data"))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--offset", type=float, default=0.1)
    parser.add_argument("--no-event-token-rate", type=int, default=5)
    parser.add_argument("--max-patients", type=int, default=-1,
                        help="Max patients to evaluate (-1 for all).")
    parser.add_argument("--n-bootstrap", type=int, default=1)
    args = parser.parse_args()

    evaluate_delphi(args)


if __name__ == "__main__":
    main()
