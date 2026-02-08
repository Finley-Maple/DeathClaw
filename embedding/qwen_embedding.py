"""
Qwen2.5 Text Embedding Extractor for Patient Descriptions

Extracts embeddings from patient text descriptions using a Qwen text model
(AutoModelForCausalLM + AutoTokenizer -- NOT the VL/vision-language variant).

Supports both:
  - Method 1: disease-before-60 natural-language summaries
  - Method 2: trajectory texts (when used as token embedder)

Refactored for eid-based I/O: reads texts keyed by eid, outputs embeddings
keyed by eid as .npz and metadata JSON.

Usage:
    python embedding/qwen_embedding.py \
        --input-csv data/preprocessed/text_before60.csv \
        --output-dir data/preprocessed/embeddings_text

    python embedding/qwen_embedding.py \
        --input-dir data/preprocessed/text_before60 \
        --output-dir data/preprocessed/embeddings_text
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""
    # Use a TEXT-ONLY model -- NOT the VL (vision-language) variant.
    # Qwen2.5-VL-* pulls in AutoProcessor -> image_transforms -> tensorflow,
    # which crashes with numpy >= 2.0.  A causal LM model avoids that entirely.
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    max_length: int = 2048
    use_8bit: bool = False
    use_4bit: bool = True
    pooling_method: str = "mean"  # "mean", "cls", "last"
    output_layer: int = -1        # Which hidden layer (-1 = last)


class QwenEmbeddingExtractor:
    """Extract embeddings from a Qwen causal-LM model."""

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the Qwen TEXT model with optional quantization."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading {self.config.model_name} on {self.config.device}...")

            quantization_config = None
            if self.config.use_4bit or self.config.use_8bit:
                from transformers import BitsAndBytesConfig
                if self.config.use_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info("Using 4-bit quantization")
                else:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("Using 8-bit quantization")

            # Tokenizer (text-only, no image processor)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Model
            model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16}
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = self.config.device

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )
            self.model.eval()
            logger.info("Model loaded successfully!")

        except ImportError as e:
            logger.error(f"Required packages not found: {e}")
            logger.error("Install: pip install transformers accelerate bitsandbytes")
            raise

    def _pool_embeddings(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling_method == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.config.pooling_method == "cls":
            return hidden_states[:, 0, :]
        elif self.config.pooling_method == "last":
            batch_size = hidden_states.shape[0]
            seq_lengths = attention_mask.sum(dim=1) - 1
            return hidden_states[torch.arange(batch_size), seq_lengths]
        else:
            raise ValueError(f"Unknown pooling method: {self.config.pooling_method}")

    def extract_embedding(self, text: str) -> np.ndarray:
        """Extract embedding for a single text string."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.config.output_layer]
            attention_mask = inputs["attention_mask"]
            pooled = self._pool_embeddings(hidden_states, attention_mask)
            return pooled.cpu().float().numpy().squeeze()

    def extract_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings for a list of texts (one at a time internally)."""
        embeddings = []
        for text in tqdm(texts, desc="Extracting embeddings"):
            embeddings.append(self.extract_embedding(text))
        return np.array(embeddings)


# ---------------------------------------------------------------------------
# EID-based patient embedding pipeline
# ---------------------------------------------------------------------------

class PatientEmbeddingProcessor:
    """Process patient texts keyed by eid and extract embeddings."""

    def __init__(
        self,
        config: EmbeddingConfig = None,
        extractor: Optional[QwenEmbeddingExtractor] = None,
    ):
        self.config = config or EmbeddingConfig()
        # Lazy init: only load the model when actually needed
        self._extractor = extractor
        self._config_for_lazy = self.config

    @property
    def extractor(self) -> QwenEmbeddingExtractor:
        if self._extractor is None:
            self._extractor = QwenEmbeddingExtractor(self._config_for_lazy)
        return self._extractor

    # -- input loaders (no model needed) ------------------------------------

    @staticmethod
    def load_from_directory(input_dir: Path) -> Dict[int, str]:
        """Load eid -> text from a directory of eid_<id>.txt files."""
        texts: Dict[int, str] = {}
        for f in sorted(input_dir.glob("eid_*.txt")):
            eid = int(f.stem.split("_", 1)[1])
            texts[eid] = f.read_text(encoding="utf-8")
        logger.info(f"Loaded {len(texts)} patient texts from {input_dir}")
        return texts

    @staticmethod
    def load_from_csv(csv_path: Path, text_col: str = "text") -> Dict[int, str]:
        """Load eid -> text from a CSV with columns [eid, <text_col>]."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        df["eid"] = df["eid"].astype(int)
        texts = dict(zip(df["eid"], df[text_col].astype(str)))
        logger.info(f"Loaded {len(texts)} patient texts from {csv_path}")
        return texts

    # -- main pipeline (model loaded on first call) -------------------------

    def process(self, texts: Dict[int, str]) -> Dict[int, np.ndarray]:
        """Extract an embedding for each eid."""
        if not texts:
            logger.warning("No texts to embed.")
            return {}
        embeddings: Dict[int, np.ndarray] = {}
        for eid, text in tqdm(texts.items(), desc="Embedding patients"):
            try:
                embeddings[eid] = self.extractor.extract_embedding(text)
            except Exception as e:
                logger.error(f"Error embedding eid {eid}: {e}")
        return embeddings

    # -- save / load --------------------------------------------------------

    @staticmethod
    def save(embeddings: Dict[int, np.ndarray], output_dir: Path, tag: str = "patient"):
        """Save embeddings as {tag}_embeddings.npz and metadata JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if not embeddings:
            logger.warning("No embeddings to save.")
            return

        npz_path = output_dir / f"{tag}_embeddings.npz"
        str_keyed = {str(eid): emb for eid, emb in embeddings.items()}
        np.savez_compressed(npz_path, **str_keyed)
        logger.info(f"Saved {len(embeddings)} embeddings to {npz_path}")

        sample_emb = next(iter(embeddings.values()))
        meta = {
            "num_patients": len(embeddings),
            "embedding_dim": int(sample_emb.shape[0]) if sample_emb.ndim > 0 else 0,
            "eids": sorted(embeddings.keys()),
        }
        meta_path = output_dir / f"{tag}_embedding_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info(f"Saved metadata to {meta_path}")

    @staticmethod
    def load(output_dir: Path, tag: str = "patient") -> Dict[int, np.ndarray]:
        """Load previously saved embeddings."""
        npz_path = output_dir / f"{tag}_embeddings.npz"
        data = np.load(npz_path)
        return {int(k): data[k] for k in data.files}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract Qwen text embeddings for patient texts (eid-based).")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input-dir", type=Path, help="Directory with eid_*.txt files.")
    grp.add_argument("--input-csv", type=Path, help="CSV with [eid, text] columns.")

    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save embeddings.")
    parser.add_argument("--tag", type=str, default="patient", help="File prefix for outputs.")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name in CSV.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID. Must be a text-only causal LM (not VL).")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls", "last"])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization.")
    args = parser.parse_args()

    # Load texts first (no model needed)
    if args.input_dir:
        texts = PatientEmbeddingProcessor.load_from_directory(args.input_dir)
    else:
        texts = PatientEmbeddingProcessor.load_from_csv(args.input_csv, text_col=args.text_col)

    if not texts:
        logger.error("No input texts found. Make sure preprocessing has been run first:")
        logger.error("  Method 1: python preprocessing/natural_text_conversion.py")
        logger.error("  Method 2: python preprocessing/generate_trajectory_text.py")
        return

    # Now load the model and embed
    config = EmbeddingConfig(
        model_name=args.model_name,
        use_4bit=not args.no_4bit,
        pooling_method=args.pooling,
        max_length=args.max_length,
    )

    processor = PatientEmbeddingProcessor(config=config)
    embeddings = processor.process(texts)
    processor.save(embeddings, args.output_dir, tag=args.tag)

    logger.info(f"Done! Processed {len(embeddings)} patients.")
    if embeddings:
        sample = next(iter(embeddings.values()))
        logger.info(f"Embedding dim: {sample.shape}")


if __name__ == "__main__":
    main()
