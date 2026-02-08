# Death Prediction Pipeline

Predict **death after age 60** using patient features and disease history before age 60, evaluated across four methods.

## Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | **Delphi** | Generative transformer for health trajectories; predicts "Death" token probability. Evaluated with DeLong AUC. |
| 2 | **Benchmarking (CoxPH)** | CoxPH survival model on binary disease features + baseline biomarkers. Evaluated with C-index and time-dependent AUC. |
| 3 | **Text Embedding + CoxPH** | Convert disease history to natural language, embed with Qwen3-Embedding, combine with baselines, fit CoxPH. |
| 4 | **Trajectory Embedding + CoxPH** | Delphi-style token + age embeddings (sin/cos), pool across events, combine with baselines, fit CoxPH. |

## Directory Structure

```
├── benchmarking/           # Survival data preprocessing & CoxPH training
│   ├── preprocess_diagnosis.py         # Extract disease features from UKB
│   ├── preprocess_survival.py          # Build survival dataset (event_flag, duration_days)
│   ├── autoprognosis_survival_dataset.csv  # Output: survival dataset
│   └── disease_before60_features.csv       # Output: binary disease flags
├── Delphi/                 # Delphi model, training & evaluation code
│   ├── model.py, train.py, utils.py    # Core Delphi code
│   └── evaluate_auc.py                 # AUC evaluation via DeLong
├── embedding/              # Embedding extraction (methods 3 & 4)
│   ├── qwen_embedding.py              # Qwen text-only embedding (method 3 tokens / method 4 texts)
│   └── trajectory_embedding.py        # Token+age embedding pipeline (method 4)
├── preprocessing/          # Preprocessing for embedding inputs
│   ├── generate_disease_trajectory.py  # Build age-at-diagnosis matrix (disease_trajectory.csv)
│   ├── generate_trajectory_text.py     # Convert matrix → Delphi-style trajectory text per patient
│   └── natural_text_conversion.py      # Convert tabular data → natural-language text per patient
├── evaluation/             # Unified evaluation & comparison
│   ├── cohort_split.py                 # Define shared train/val/test split
│   ├── evaluate_delphi.py              # Evaluate Delphi on shared cohort
│   ├── evaluate_benchmarking.py        # Train & evaluate CoxPH on shared cohort
│   ├── evaluate_embedding_survival.py  # Train & evaluate CoxPH on embeddings
│   └── unified_evaluation.py           # Compare all methods in one table
├── data/                   # Raw & processed data (gitignored)
├── UKB_extraction/         # UK Biobank data extraction tools
├── docs/                   # Proposals, references
└── run_pipeline.sh         # One-command pipeline runner (steps 1–7)
```

## Quick Start (one command)

Run the entire pipeline end-to-end with `run_pipeline.sh`:

```bash
# Local / CPU: 10k sample, Qwen3-Embedding-0.6B (auto-selected)
bash run_pipeline.sh

# Full dataset, auto-selects model based on device
bash run_pipeline.sh --full

# GPU server: full dataset, 8B model
bash run_pipeline.sh --full --embedding-model Qwen/Qwen3-Embedding-8B

# Mid-range GPU: 4B model
bash run_pipeline.sh --full --embedding-model Qwen/Qwen3-Embedding-4B

# Skip preprocessing if data already exists
bash run_pipeline.sh --skip-preprocess --steps 5,6,7

# Skip Delphi (if no checkpoint available)
bash run_pipeline.sh --skip-delphi
```

Options:

| Flag | Description |
|------|-------------|
| `--full` | Use all participants instead of a 10k sample |
| `--sample-size N` | Custom sample size (default: 10000) |
| `--embedding-model MODEL` | Qwen3-Embedding-0.6B/4B/8B (auto-selected by device) |
| `--token-mode random\|qwen` | Trajectory token embedding mode (default: random) |
| `--skip-preprocess` | Skip steps 1-2 if CSV files already exist |
| `--skip-delphi` | Skip Delphi evaluation |
| `--steps 1,2,3,...` | Run only specific steps |
| `--device cuda\|cpu` | Force device (auto-detected by default) |
| `--random-state N` | Random seed (default: 42) |

The script logs everything to `pipeline_YYYYMMDD_HHMMSS.log` and prints the comparison table at the end.

## Pipeline (step by step)

### Step 0: UKB data extraction

Extract raw UK Biobank data into `data/`. See `UKB_extraction/` for tooling.

### Step 1: Build survival dataset & disease features

These scripts produce the two CSV files that all downstream steps depend on.

```bash
python benchmarking/preprocess_diagnosis.py    # → disease_before60_features.csv
python benchmarking/preprocess_survival.py     # → autoprognosis_survival_dataset.csv (10k sample)
# Or use the full dataset:
python benchmarking/preprocess_survival.py --all
```

### Step 2: Build disease trajectory matrix

Generate per-patient age-at-diagnosis for all diseases (needed by method 4).

```bash
python preprocessing/generate_disease_trajectory.py   # → data/preprocessed/disease_trajectory.csv
```

### Step 3: Define shared cohort split

Create a single train/val/test split (70/15/15, stratified) used by all methods.

```bash
python evaluation/cohort_split.py              # → evaluation/cohort_split.json
```

### Step 4: Generate embedding inputs

**Method 3 (text embedding):** convert tabular data to natural-language summaries.

```bash
python preprocessing/natural_text_conversion.py \
    --output-csv  data/preprocessed/text_before60.csv \
    --output-dir  data/preprocessed/text_before60
```

**Method 4 (trajectory embedding):** convert trajectory matrix to Delphi-style text.

```bash
python preprocessing/generate_trajectory_text.py \
    --output-csv  data/preprocessed/trajectory_before60.csv \
    --output-dir  data/preprocessed/trajectory_before60
```

### Step 5: Compute embeddings

**Method 3:** embed natural-language texts with Qwen3-Embedding.

```bash
# GPU server (8B, 4096-dim):
python embedding/qwen_embedding.py \
    --input-csv   data/preprocessed/text_before60.csv \
    --output-dir  data/preprocessed/embeddings_text \
    --model-name  Qwen/Qwen3-Embedding-8B

# Local / CPU (0.6B, 1024-dim):
python embedding/qwen_embedding.py \
    --input-csv   data/preprocessed/text_before60.csv \
    --output-dir  data/preprocessed/embeddings_text \
    --model-name  Qwen/Qwen3-Embedding-0.6B \
    --no-flash-attn
```

**Method 4:** embed trajectory token+age vectors.

```bash
# Random token embeddings (CPU, for testing):
python embedding/trajectory_embedding.py \
    --input-csv   data/preprocessed/trajectory_before60.csv \
    --output-dir  data/preprocessed/embeddings_traj

# Or with Qwen token embeddings (GPU):
python embedding/trajectory_embedding.py \
    --input-csv   data/preprocessed/trajectory_before60.csv \
    --output-dir  data/preprocessed/embeddings_traj \
    --token-mode  qwen
```

### Step 6: Train & evaluate each method

Each evaluation script trains on the shared train split and evaluates on val/test.

```bash
# Method 1: Delphi
python evaluation/evaluate_delphi.py

# Method 2: Benchmarking (CoxPH on binary disease features)
python evaluation/evaluate_benchmarking.py

# Method 3: Text Embedding + CoxPH
python evaluation/evaluate_embedding_survival.py \
    --embedding-dir data/preprocessed/embeddings_text \
    --tag patient \
    --method-name text_embedding

# Method 4: Trajectory Embedding + CoxPH
python evaluation/evaluate_embedding_survival.py \
    --embedding-dir data/preprocessed/embeddings_traj \
    --tag trajectory \
    --method-name trajectory_embedding
```

### Step 7: Unified comparison

```bash
python evaluation/unified_evaluation.py        # → evaluation/unified_comparison.csv
```

## Requirements

- Python 3.9+
- Core: `numpy`, `pandas`, `lifelines`, `scikit-survival`, `tqdm`
- Delphi: see `Delphi/requirements.txt`
- Qwen3-Embedding: see `embedding/requirements_qwen.txt` (`transformers>=4.51.0`, `torch`; 0.6B runs on CPU, 8B needs GPU)