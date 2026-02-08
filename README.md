# Death Prediction Pipeline

Predict **death after age 60** using patient features and disease history before age 60, evaluated across four methods.

## Methods

| # | Method | Description |
|---|--------|-------------|
| 1 | **Delphi** | Generative transformer model for health trajectories; predicts "Death" token probability |
| 2 | **Benchmarking (survival)** | AutoPrognosis survival models using binary disease features + baseline biomarkers |
| 3 | **Text embedding** | Convert disease history to natural language, embed with Qwen, feed to survival model |
| 4 | **Trajectory embedding** | Delphi-style token + age embeddings, pooled across events, feed to survival model |

## Directory Structure

```
├── benchmarking/           # AutoPrognosis survival pipeline
├── Delphi/                 # Delphi model, evaluation, training code
├── embedding/              # Qwen text embedding & trajectory embedding
├── preprocessing/          # Disease trajectory generation, natural text conversion
├── evaluation/             # Unified evaluation, cohort split, comparison
├── data/                   # Raw / processed data (gitignored)
├── UKB_extraction/         # UK Biobank data extraction tools
└── docs/                   # Proposals, references
```

## Quick Start

1. **Preprocessing**: Generate disease features and trajectories
   ```bash
   python benchmarking/preprocess_diagnosis.py
   python benchmarking/preprocess_survival.py
   python preprocessing/generate_disease_trajectory.py
   ```

2. **Cohort split**: Define shared train/val/test split
   ```bash
   python evaluation/cohort_split.py
   ```

3. **Embeddings**: Generate text and trajectory embeddings
   ```bash
   python embedding/qwen_embedding.py --input-dir data/preprocessed/text_before60 --output-dir data/preprocessed/embeddings_text
   python embedding/trajectory_embedding.py --input-dir data/preprocessed/trajectory_before60 --output-dir data/preprocessed/embeddings_traj
   ```

4. **Evaluation**: Run unified evaluation
   ```bash
   python evaluation/unified_evaluation.py
   ```

## Requirements

- Python 3.9+
- See `Delphi/requirements.txt` and `embedding/requirements_qwen.txt` for model-specific dependencies
- AutoPrognosis: `pip install autoprognosis`
