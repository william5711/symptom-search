"""Sentence embeddings with an on-disk cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


class Encoder:
    """Turns text into unit-length vectors, so a dot product is cosine similarity."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Path = DEFAULT_CACHE_DIR):
        # Imported here because torch is slow to load and tests can stub this class.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    def encode_cached(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a fixed corpus, reusing the saved vectors if neither it nor the model changed."""
        key = hashlib.sha256("\n".join([self.model_name, *texts]).encode("utf-8")).hexdigest()[:16]
        path = self.cache_dir / f"embeddings-{key}.npy"
        if path.exists():
            return np.load(path)

        vectors = self.encode(texts)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of `a` and each row of `b` (both unit length)."""
    # NumPy 2.0 on Apple Silicon raises false divide-by-zero warnings from matmul.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a @ b.T
