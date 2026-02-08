"""
Delphi evaluation adapter for the shared cohort.

Uses the `evaluate_auc_pipeline()` function from Delphi/evaluate_auc.py
as the canonical evaluation backend.

This script:
  1. Loads the Delphi model checkpoint.
  2. Loads the test.bin (or val.bin) aligned with cohort_split.json.
  3. Runs evaluate_auc_pipeline for the Death token.
  4. Saves results to evaluation/delphi_results/.

NOTE: Delphi binary data must be generated first using:
    python Delphi/preprocess_delphi_binary.py

Usage:
    python evaluation/evaluate_delphi.py \
        --ckpt-path Delphi/Delphi-2M-respiratory/ckpt.pt \
        --data-path Delphi/data/ukb_respiratory_data \
        --split test \
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

    # Load evaluation data
    # Prefer test.bin for final evaluation; fall back to val.bin
    data_dir = Path(args.data_path)
    split = args.split
    bin_path = data_dir / f"{split}.bin"

    if not bin_path.exists():
        # Try fallback
        fallbacks = ["test.bin", "val.bin"]
        found = False
        for fb in fallbacks:
            candidate = data_dir / fb
            if candidate.exists():
                print(f"  {bin_path} not found, using {candidate} instead.")
                bin_path = candidate
                split = fb.replace(".bin", "")
                found = True
                break

        if not found:
            print(f"\nERROR: No binary data found in {data_dir}/")
            print(f"  Expected: {data_dir}/test.bin or {data_dir}/val.bin")
            print(f"\n  Run Delphi preprocessing first:")
            print(f"    python Delphi/preprocess_delphi_binary.py")
            print(f"\n  This will generate train.bin, val.bin, test.bin")
            print(f"  aligned with the shared cohort_split.json.")
            sys.exit(1)

    print(f"Loading {split} data from {bin_path}...")
    data = np.fromfile(str(bin_path), dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    data_p2i = get_p2i(data)

    n_patients = len(data_p2i)
    if args.max_patients > 0:
        n_patients = min(n_patients, args.max_patients)

    print(f"Evaluating on {n_patients} patients ({split} split)...")
    d_batch = get_batch(
        range(n_patients),
        data,
        data_p2i,
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
    # Exclude ICD codes about death (e.g. "O96 Death from ...") - only want the
    # pure "Death" label
    death_rows_exact = death_rows[death_rows["name"].str.strip() == "Death"]
    if len(death_rows_exact) > 0:
        death_token_ids = death_rows_exact["index"].tolist()
    elif len(death_rows) > 0:
        death_token_ids = death_rows["index"].tolist()
    else:
        # Fallback to known index
        death_token_ids = [1269]
    print(f"Death token indices: {death_token_ids}")

    # Run evaluation
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
        meta_info={"method": "delphi", "cohort": "shared", "split": split},
    )

    print(f"\n=== Delphi Death Token AUC Results ({split} split) ===")
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
        "split": split,
        "n_patients": n_patients,
        "death_token_ids": death_token_ids,
        "results": death_results.to_dict("records") if len(death_results) > 0 else [],
    }
    summary_path = output_path / "delphi_death_summary.json"
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
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Which split to evaluate on (default: test)")
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
