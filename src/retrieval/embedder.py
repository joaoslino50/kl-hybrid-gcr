"""SentenceBERT encoder for questions, KG entities, relations, and verbalized triples."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Sequence

import torch
import torch.nn.functional as F

from src.utils.device import get_device

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class GraphEmbedder:
    """Wrapper around `sentence-transformers` with lazy loading and deterministic mock mode."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: torch.device | None = None,
        batch_size: int = 64,
        use_mock: bool = False,
        mock_dim: int = 128,
    ) -> None:
        """Initialize the SentenceBERT graph embedder.

        Args:
            model_name: Hugging Face `sentence-transformers` model identifier.
            device: Compute device; defaults to `get_device()`.
            batch_size: Encoding mini-batch size.
            use_mock: Use lightweight lexical-hash embeddings without downloading model weights.
            mock_dim: Embedding dimensionality when `use_mock=True`.
        """
        self.model_name = model_name
        self.device = device if device is not None else get_device()
        self.batch_size = batch_size
        self.use_mock = use_mock
        self.mock_dim = mock_dim
        self._model = None

    def _load_model(self):
        """Lazily instantiate `SentenceTransformer` on the configured device."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceBERT model '%s' on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=str(self.device))
        return self._model

    def _encode_mock(self, texts: Sequence[str]) -> torch.Tensor:
        """Produce deterministic L2-normalized hashed lexical embeddings for offline testing."""
        matrix = torch.zeros((len(texts), self.mock_dim), dtype=torch.float32)
        for row_idx, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower()) or ["_empty_"]
            for tok in tokens:
                digest = hashlib.sha256(tok.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.mock_dim
                sign = 1.0 if (digest[4] % 2 == 0) else -1.0
                matrix[row_idx, bucket] += sign
            if torch.all(matrix[row_idx] == 0):
                matrix[row_idx, 0] = 1.0
        normalized = F.normalize(matrix, p=2, dim=-1)
        return normalized.to(self.device)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode a sequence of strings into L2-normalized dense embeddings `[N, D]`."""
        if not texts:
            dim = self.mock_dim if self.use_mock else 384
            return torch.empty((0, dim), dtype=torch.float32, device=self.device)

        if self.use_mock:
            return self._encode_mock(texts)

        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.to(self.device, dtype=torch.float32)

    def encode_question(self, question: str) -> torch.Tensor:
        """Encode a single question string into a 1D L2-normalized tensor `[D]`."""
        return self.encode_texts([question])[0]

    def compute_similarities(self, question: str, candidates: Sequence[str]) -> torch.Tensor:
        """Compute cosine similarities `[N]` between `question` and `candidates`."""
        if not candidates:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        q_emb = self.encode_question(question)  # [D]
        c_embs = self.encode_texts(candidates)  # [N, D]
        return torch.mv(c_embs, q_emb)

