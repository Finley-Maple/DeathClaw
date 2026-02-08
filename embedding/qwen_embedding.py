"""
Qwen3-Embedding Extractor for Patient Descriptions

Uses Qwen3-Embedding models (dedicated embedding models, NOT generative LMs)
to extract text embeddings for patient descriptions.

Available models (pick based on hardware):
  - Qwen/Qwen3-Embedding-0.6B  (1024-dim, ~1.2 GB, runs on CPU / laptop)
  - Qwen/Qwen3-Embedding-4B    (2560-dim, ~8 GB,  mid-range GPU)
  - Qwen/Qwen3-Embedding-8B    (4096-dim, ~16 GB, GPU recommended)

All models support Matryoshka Representation Learning (MRL), so you can
truncate the embedding to any dimension <= max via --embedding-dim.

Requires: transformers>=4.51.0, torch

Usage:
    # Local / CPU (0.6B model):
    python embedding/qwen_embedding.py \
        --input-csv data/preprocessed/text_before60.csv \
        --output-dir data/preprocessed/embeddings_text \
        --model-name Qwen/Qwen3-Embedding-0.6B

    # GPU server (8B model):
    python embedding/qwen_embedding.py \
        --input-csv data/preprocessed/text_before60.csv \
        --output-dir data/preprocessed/embeddings_text \
        --model-name Qwen/Qwen3-Embedding-8B
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default embedding dimensions per model (max native dim)
MODEL_DEFAULT_DIM = {
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
}


# ---------------------------------------------------------------------------
# Pooling (last-token pool as recommended by Qwen3-Embedding)
# ---------------------------------------------------------------------------

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Extract the embedding from the last non-padding token.

    This is the recommended pooling method for Qwen3-Embedding models.
    With left-padding (padding_side='left'), the last token is always
    the final token in the sequence.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """Configuration for Qwen3-Embedding extraction."""
    # Model selection:
    #   GPU:   Qwen/Qwen3-Embedding-8B  (best quality, ~16 GB)
    #   Mid:   Qwen/Qwen3-Embedding-4B  (~8 GB)
    #   Local: Qwen/Qwen3-Embedding-0.6B (~1.2 GB, CPU-friendly)
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 8192
    batch_size: int = 4
    # MRL: truncate output to this many dims (0 = use full native dim)
    embedding_dim: int = 0
    # Instruction prefix for embedding (improves retrieval 1-5%)
    instruction: str = ""
    # Whether to L2-normalize embeddings
    normalize: bool = True
    # Use flash_attention_2 for faster inference on GPU
    use_flash_attn: bool = True


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class QwenEmbeddingExtractor:
    """Extract embeddings using a Qwen3-Embedding model."""

    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the Qwen3-Embedding model."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            logger.error(f"transformers not found: {e}")
            logger.error("Install: pip install 'transformers>=4.51.0' torch accelerate")
            raise

        logger.info(f"Loading {self.config.model_name} on {self.config.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side="left",
            trust_remote_code=True,
        )

        model_kwargs = {"trust_remote_code": True}

        # Use float16 on GPU, float32 on CPU
        if self.config.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            if self.config.use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModel.from_pretrained(
            self.config.model_name, **model_kwargs
        )
        self.model.eval()

        if self.config.device != "cuda" or "device_map" not in model_kwargs:
            self.model = self.model.to(self.config.device)

        native_dim = MODEL_DEFAULT_DIM.get(self.config.model_name, "unknown")
        logger.info(f"Model loaded! Native embedding dim: {native_dim}")

    def _get_target_dim(self) -> Optional[int]:
        """Return the target embedding dimension, or None for full native dim."""
        if self.config.embedding_dim > 0:
            return self.config.embedding_dim
        return None

    def extract_embedding(self, text: str) -> np.ndarray:
        """Extract embedding for a single text string."""
        return self.extract_batch_embeddings([text])[0]

    def extract_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings for a batch of texts."""
        # Prepend instruction if set
        if self.config.instruction:
            texts = [
                f"Instruct: {self.config.instruction}\nQuery: {t}"
                for t in texts
            ]

        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )

            # MRL: truncate to target dim if specified
            target_dim = self._get_target_dim()
            if target_dim is not None:
                embeddings = embeddings[:, :target_dim]

            # L2 normalize
            if self.config.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().float().numpy()


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
        self._extractor = extractor
        self._config_for_lazy = self.config

    @property
    def extractor(self) -> QwenEmbeddingExtractor:
        """Lazy-load the model on first use."""
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
        """Extract an embedding for each eid, batched for efficiency."""
        if not texts:
            logger.warning("No texts to embed.")
            return {}

        eids = list(texts.keys())
        all_texts = [texts[eid] for eid in eids]
        batch_size = self.config.batch_size
        embeddings: Dict[int, np.ndarray] = {}

        for i in tqdm(range(0, len(all_texts), batch_size), desc="Embedding patients"):
            batch_eids = eids[i : i + batch_size]
            batch_texts = all_texts[i : i + batch_size]
            try:
                batch_embs = self.extractor.extract_batch_embeddings(batch_texts)
                for eid, emb in zip(batch_eids, batch_embs):
                    embeddings[eid] = emb
            except Exception as e:
                logger.error(f"Error embedding batch starting at index {i}: {e}")
                # Fall back to one-by-one
                for eid, text in zip(batch_eids, batch_texts):
                    try:
                        embeddings[eid] = self.extractor.extract_embedding(text)
                    except Exception as e2:
                        logger.error(f"Error embedding eid {eid}: {e2}")

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
    parser = argparse.ArgumentParser(
        description="Extract Qwen3-Embedding text embeddings for patient texts (eid-based).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model choices (pick based on hardware):
  Qwen/Qwen3-Embedding-0.6B   1024-dim, ~1.2 GB  (CPU / laptop)
  Qwen/Qwen3-Embedding-4B     2560-dim, ~8 GB    (mid-range GPU)
  Qwen/Qwen3-Embedding-8B     4096-dim, ~16 GB   (GPU server)
""",
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input-dir", type=Path, help="Directory with eid_*.txt files.")
    grp.add_argument("--input-csv", type=Path, help="CSV with [eid, text] columns.")

    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save embeddings.")
    parser.add_argument("--tag", type=str, default="patient", help="File prefix for outputs.")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name in CSV.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="HuggingFace model ID (Qwen3-Embedding-0.6B/4B/8B).")
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Max token length (Qwen3-Embedding supports up to 32k).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for embedding extraction.")
    parser.add_argument("--embedding-dim", type=int, default=0,
                        help="Custom output dim via MRL (0 = full native dim).")
    parser.add_argument("--instruction", type=str, default="",
                        help="Task instruction prefix (e.g. 'Represent the clinical record').")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip L2 normalization of embeddings.")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Disable flash_attention_2.")
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

    # Configure and run
    config = EmbeddingConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        instruction=args.instruction,
        normalize=not args.no_normalize,
        use_flash_attn=not args.no_flash_attn,
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
