"""
Trajectory Embedding Module (Method 2)

Embeds per-patient disease trajectories by:
  1. Token embedding: embed each unique disease string with Qwen (or a lookup).
  2. Age embedding: sin/cos encoding of age (same scheme as Delphi AgeEncoding).
  3. Combine: concat(age_embedding, token_embedding) per event, then pool over events.
  4. Output by eid in the same format as method 1 for downstream survival.

Usage:
    # Using Qwen for token embeddings (requires GPU + model download):
    python embedding/trajectory_embedding.py \
        --input-csv  data/preprocessed/trajectory_before60.csv \
        --output-dir data/preprocessed/embeddings_traj

    # Using cached token embeddings:
    python embedding/trajectory_embedding.py \
        --input-csv  data/preprocessed/trajectory_before60.csv \
        --output-dir data/preprocessed/embeddings_traj \
        --token-cache data/preprocessed/token_embeddings_cache.npz
"""

import argparse
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PANDAS_NO_IMPORT_NUMEXPR", "1")
os.environ.setdefault("PANDAS_NO_IMPORT_BOTTLENECK", "1")

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Age Encoding (mirrors Delphi/model.py AgeEncoding)
# ---------------------------------------------------------------------------

class AgeEncoder:
    """
    Sin/cos age encoding matching Delphi's AgeEncoding.

    The original Delphi uses:
        div_term = exp(arange(0, n_embd, 2) * (-log(10000) / n_embd))
        y[..., 0::2] = sin(age / 365.25 * div_term)
        y[..., 1::2] = cos(age / 365.25 * div_term)

    Here age is in *years* (not days), so we multiply by 365.25 first
    to stay compatible with Delphi (which stores age in days).
    """

    def __init__(self, n_embd: int = 128):
        self.n_embd = n_embd
        self.div_term = np.exp(
            np.arange(0, n_embd, 2) * (-math.log(10000.0) / n_embd)
        )

    def encode(self, age_years: float) -> np.ndarray:
        """Encode a single age (in years) to an n_embd-dim vector."""
        age_days = age_years * 365.25
        y = np.zeros(self.n_embd, dtype=np.float32)
        y[0::2] = np.sin(age_days / 365.25 * self.div_term)
        y[1::2] = np.cos(age_days / 365.25 * self.div_term)
        return y

    def encode_batch(self, ages: np.ndarray) -> np.ndarray:
        """Encode an array of ages -> (len(ages), n_embd)."""
        ages_days = ages * 365.25
        n = len(ages)
        y = np.zeros((n, self.n_embd), dtype=np.float32)
        y[:, 0::2] = np.sin(np.outer(ages_days, self.div_term) / 365.25)
        y[:, 1::2] = np.cos(np.outer(ages_days, self.div_term) / 365.25)
        return y


# ---------------------------------------------------------------------------
# Token Embedding
# ---------------------------------------------------------------------------

class TokenEmbedder:
    """
    Embeds disease/event tokens.

    Two modes:
      - qwen: use QwenEmbeddingExtractor to embed each unique token string.
      - random: fixed random embeddings per unique token (for testing).

    Embeddings are cached so each unique token is embedded only once.
    """

    def __init__(
        self,
        mode: str = "random",
        token_dim: int = 128,
        cache_path: Optional[Path] = None,
        qwen_extractor=None,
    ):
        self.mode = mode
        self.token_dim = token_dim
        self.cache: Dict[str, np.ndarray] = {}
        self.qwen_extractor = qwen_extractor

        if cache_path and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            self.cache = {str(k): data[k] for k in data.files}
            logger.info(f"Loaded {len(self.cache)} cached token embeddings from {cache_path}")

    def embed(self, token: str) -> np.ndarray:
        if token in self.cache:
            return self.cache[token]

        if self.mode == "qwen":
            if self.qwen_extractor is None:
                raise RuntimeError("Qwen extractor not provided for mode='qwen'")
            emb = self.qwen_extractor.extract_embedding(token)
            # Project to token_dim if needed
            if emb.shape[0] != self.token_dim:
                # Simple truncation or zero-padding
                result = np.zeros(self.token_dim, dtype=np.float32)
                n = min(self.token_dim, emb.shape[0])
                result[:n] = emb[:n]
                emb = result
        elif self.mode == "random":
            rng = np.random.RandomState(hash(token) % (2 ** 31))
            emb = rng.randn(self.token_dim).astype(np.float32) * 0.02
        else:
            raise ValueError(f"Unknown token embedding mode: {self.mode}")

        self.cache[token] = emb
        return emb

    def save_cache(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.cache)
        logger.info(f"Saved {len(self.cache)} token embeddings to {path}")


# ---------------------------------------------------------------------------
# Trajectory Parser
# ---------------------------------------------------------------------------

def parse_trajectory(text: str) -> List[Tuple[float, str]]:
    """
    Parse a Delphi-style trajectory text into (age, event) pairs.

    Lines like "20.0: G43 Migraine" become (20.0, "G43 Migraine").
    Lines like "Before 60: Depression" (fallback format) also parsed.
    """
    events = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([\d.]+):\s*(.+)$", line)
        if m:
            age = float(m.group(1))
            event = m.group(2).strip()
            events.append((age, event))
        else:
            # Fallback: "Before 60: ..." -> use age 30 as midpoint
            m2 = re.match(r"^Before \d+:\s*(.+)$", line)
            if m2:
                events.append((30.0, m2.group(1).strip()))
    return events


# ---------------------------------------------------------------------------
# Main embedding pipeline
# ---------------------------------------------------------------------------

class TrajectoryEmbeddingPipeline:
    """
    Full pipeline: parse trajectory text -> embed tokens + ages -> pool -> output.
    """

    def __init__(
        self,
        age_dim: int = 128,
        token_dim: int = 128,
        token_mode: str = "random",
        pooling: str = "mean",
        cache_path: Optional[Path] = None,
        qwen_extractor=None,
    ):
        self.age_encoder = AgeEncoder(n_embd=age_dim)
        self.token_embedder = TokenEmbedder(
            mode=token_mode, token_dim=token_dim,
            cache_path=cache_path, qwen_extractor=qwen_extractor,
        )
        self.pooling = pooling
        self.output_dim = age_dim + token_dim

    def embed_patient(self, trajectory_text: str) -> np.ndarray:
        """
        Embed a single patient's trajectory.

        Returns a 1-D vector of shape (age_dim + token_dim,).
        """
        events = parse_trajectory(trajectory_text)

        if not events:
            return np.zeros(self.output_dim, dtype=np.float32)

        # Skip "No event" tokens for the embedding (they carry no disease info)
        informative = [(age, tok) for age, tok in events if tok != "No event"]
        if not informative:
            # Only "No event" and possibly sex token
            informative = events[:1]  # keep at least the sex token

        ages = np.array([e[0] for e in informative], dtype=np.float32)
        age_embs = self.age_encoder.encode_batch(ages)  # (n, age_dim)
        token_embs = np.array(
            [self.token_embedder.embed(e[1]) for e in informative],
            dtype=np.float32,
        )  # (n, token_dim)

        combined = np.concatenate([age_embs, token_embs], axis=1)  # (n, age_dim+token_dim)

        if self.pooling == "mean":
            return combined.mean(axis=0)
        elif self.pooling == "max":
            return combined.max(axis=0)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def process_cohort(
        self,
        texts: Dict[int, str],
        output_dir: Optional[Path] = None,
        tag: str = "trajectory",
    ) -> Dict[int, np.ndarray]:
        """
        Embed all patients.

        Args:
            texts: {eid: trajectory_text}
            output_dir: If given, save .npz + metadata.
            tag: File prefix.

        Returns:
            {eid: embedding_vector}
        """
        from tqdm import tqdm

        embeddings: Dict[int, np.ndarray] = {}
        for eid, text in tqdm(texts.items(), desc="Embedding trajectories"):
            try:
                embeddings[eid] = self.embed_patient(text)
            except Exception as e:
                logger.error(f"Error embedding eid {eid}: {e}")

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            npz_path = output_dir / f"{tag}_embeddings.npz"
            str_keyed = {str(eid): emb for eid, emb in embeddings.items()}
            np.savez_compressed(npz_path, **str_keyed)
            logger.info(f"Saved {len(embeddings)} embeddings to {npz_path}")

            sample = next(iter(embeddings.values())) if embeddings else np.zeros(0)
            meta = {
                "num_patients": len(embeddings),
                "embedding_dim": int(sample.shape[0]) if sample.ndim > 0 else 0,
                "age_dim": self.age_encoder.n_embd,
                "token_dim": self.token_embedder.token_dim,
                "pooling": self.pooling,
                "eids": sorted(embeddings.keys()),
            }
            meta_path = output_dir / f"{tag}_embedding_metadata.json"
            meta_path.write_text(json.dumps(meta, indent=2))

            # Save token cache
            cache_path = output_dir / f"{tag}_token_cache.npz"
            self.token_embedder.save_cache(cache_path)

        return embeddings

    @staticmethod
    def load(output_dir: Path, tag: str = "trajectory") -> Dict[int, np.ndarray]:
        npz_path = output_dir / f"{tag}_embeddings.npz"
        data = np.load(npz_path)
        return {int(k): data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Helpers for loading input
# ---------------------------------------------------------------------------

def load_texts_from_csv(csv_path: Path, text_col: str = "trajectory_text") -> Dict[int, str]:
    df = pd.read_csv(csv_path)
    df["eid"] = df["eid"].astype(int)
    return dict(zip(df["eid"], df[text_col].astype(str)))


def load_texts_from_dir(input_dir: Path) -> Dict[int, str]:
    texts = {}
    for f in sorted(input_dir.glob("eid_*.txt")):
        eid = int(f.stem.split("_", 1)[1])
        texts[eid] = f.read_text(encoding="utf-8")
    return texts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trajectory token+age embedding (method 2).")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input-csv", type=Path, help="CSV with [eid, trajectory_text].")
    grp.add_argument("--input-dir", type=Path, help="Directory with eid_*.txt files.")

    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save embeddings.")
    parser.add_argument("--tag", type=str, default="trajectory", help="File prefix.")
    parser.add_argument("--age-dim", type=int, default=128, help="Age embedding dimension.")
    parser.add_argument("--token-dim", type=int, default=128, help="Token embedding dimension.")
    parser.add_argument("--token-mode", type=str, default="random",
                        choices=["random", "qwen"], help="How to embed tokens.")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--token-cache", type=Path, default=None,
                        help="Path to cached token embeddings (.npz).")
    parser.add_argument("--text-col", type=str, default="trajectory_text")
    args = parser.parse_args()

    # Optionally initialize Qwen for token embedding
    qwen_extractor = None
    if args.token_mode == "qwen":
        from qwen_embedding import QwenEmbeddingExtractor, EmbeddingConfig
        qwen_extractor = QwenEmbeddingExtractor(EmbeddingConfig(use_4bit=True))

    pipeline = TrajectoryEmbeddingPipeline(
        age_dim=args.age_dim,
        token_dim=args.token_dim,
        token_mode=args.token_mode,
        pooling=args.pooling,
        cache_path=args.token_cache,
        qwen_extractor=qwen_extractor,
    )

    if args.input_csv:
        texts = load_texts_from_csv(args.input_csv, text_col=args.text_col)
    else:
        texts = load_texts_from_dir(args.input_dir)

    logger.info(f"Loaded {len(texts)} trajectory texts.")
    embeddings = pipeline.process_cohort(texts, output_dir=args.output_dir, tag=args.tag)
    logger.info(f"Done! {len(embeddings)} patients embedded.")


if __name__ == "__main__":
    main()
