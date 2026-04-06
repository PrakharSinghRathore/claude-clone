"""
Lightweight Embedding Engine via Character N-Gram Hashing.

Implements a fastText-inspired embedding approach that converts arbitrary
text into fixed-size float vectors using character n-gram hashing.  The
engine uses only the Python standard library (``math``, ``hashlib``,
``collections``) — no numpy, no sentence-transformers.

Approach
--------
1. **Tokenisation** — split text into whitespace-delimited tokens.
2. **Character n-gram extraction** — for each token, generate character
   3-grams, 4-grams, and 5-grams (with boundary markers ``<`` and ``>``).
3. **Hashing** — each n-gram is hashed into one of *DIM* buckets using
   Python's built-in ``hash()`` function.
4. **TF weighting** — bucket values are weighted by the token frequency
   of the originating token (with sub-linear scaling).
5. **L2 normalisation** — the final vector is normalised to unit length
   so that cosine similarity reduces to a dot product.

This produces vectors that capture sub-word morphological similarity:
e.g. ``"running"`` and ``"runs"`` share many character n-grams and will
have high cosine similarity even though they differ as whole tokens.

Usage::

    engine = EmbeddingEngine(dim=256)
    vec = engine.encode("How do I deploy to AWS ECS?")
    sim = engine.cosine_similarity(vec, engine.encode("Cloud deployment guide"))
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Default embedding dimensionality (number of hash buckets).
DEFAULT_DIM: int = 256

#: Character n-gram sizes to extract from each token.
NGRAM_SIZES: Tuple[int, ...] = (3, 4, 5)

#: Boundary marker prepended / appended to tokens before n-gram extraction
#: (mirrors fastText's approach).
_BOUNDARY: str = "<"

#: Sub-linear TF scaling factor:  ``tf_scaled = 1 + log(tf)``  when ``tf > 0``.
_TF_LOG_SCALE: float = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingEngine
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """Lightweight hash-based text embedding engine.

    Converts text into fixed-size dense vectors using character n-gram
    hashing.  The resulting vectors are L2-normalised so that cosine
    similarity is equivalent to the dot product.

    Parameters
    ----------
    dim:
        Dimensionality of the output vectors (number of hash buckets).
        Default is 256.  Must be a positive integer.

    Examples
    --------
    >>> engine = EmbeddingEngine(dim=128)
    >>> vec = engine.encode("hello world")
    >>> len(vec)
    128
    >>> abs(sum(v * v for v in vec) - 1.0) < 1e-6  # unit vector
    True
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {dim}")
        self.dim: int = dim

    # ── Public API ──────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[float]:
        """Encode a single text string into a dense float vector.

        The vector is L2-normalised to unit length.

        Parameters
        ----------
        text:
            The input text to embed.

        Returns
        -------
        list[float]
            A list of ``self.dim`` floats representing the embedding.
        """
        vec = self._compute_raw_vector(text)
        return self._l2_normalise(vec)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple text strings into dense float vectors.

        Parameters
        ----------
        texts:
            A list of input text strings.

        Returns
        -------
        list[list[float]]
            A list of embedding vectors, one per input text.
        """
        return [self.encode(text) for text in texts]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute the cosine similarity between two vectors.

        Both vectors should ideally be L2-normalised (as produced by
        :meth:`encode`).  If either vector has zero magnitude the result
        is ``0.0``.

        Parameters
        ----------
        a:
            First vector.
        b:
            Second vector.  Must have the same length as *a*.

        Returns
        -------
        float
            Cosine similarity in the range [0.0, 1.0].

        Raises
        ------
        ValueError
            If the vectors have different lengths.
        """
        if len(a) != len(b):
            raise ValueError(
                f"Vector length mismatch: {len(a)} vs {len(b)}"
            )

        dot = 0.0
        mag_a = 0.0
        mag_b = 0.0
        for va, vb in zip(a, b):
            dot += va * vb
            mag_a += va * va
            mag_b += vb * vb

        denom = math.sqrt(mag_a) * math.sqrt(mag_b)
        if denom == 0.0:
            return 0.0

        return min(max(dot / denom, 0.0), 1.0)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase and split text on whitespace into raw tokens.

        Punctuation is stripped from token boundaries.  Tokens shorter
        than 2 characters are dropped.

        Parameters
        ----------
        text:
            Raw input text.

        Returns
        -------
        list[str]
            Lowercased tokens ready for n-gram extraction.
        """
        tokens: List[str] = []
        for word in text.lower().split():
            # Strip common punctuation from edges.
            stripped = word.strip(".,;:!?\"'()[]{}<>-_+=/\\|~`@#$%^&*")
            if len(stripped) >= 2:
                tokens.append(stripped)
        return tokens

    def _extract_ngrams(self, token: str) -> List[str]:
        """Extract character n-grams from a single token.

        Boundary markers are prepended and appended to the token (e.g.
        ``"cat"`` becomes ``"<cat>"``) before sliding-window extraction.

        Parameters
        ----------
        token:
            A single token string.

        Returns
        -------
        list[str]
            Character n-grams of sizes defined in :data:`NGRAM_SIZES`.
        """
        padded = _BOUNDARY + token + _BOUNDARY
        ngrams: List[str] = []
        for n in NGRAM_SIZES:
            for i in range(len(padded) - n + 1):
                ngrams.append(padded[i : i + n])
        return ngrams

    def _hash_ngram(self, ngram: str) -> int:
        """Hash an n-gram string into a bucket index in ``[0, self.dim)``.

        Uses Python's built-in ``hash()`` which is fast and has good
        distribution.  The result is taken modulo ``self.dim`` to map
        into the fixed number of buckets.

        Parameters
        ----------
        ngram:
            A character n-gram string.

        Returns
        -------
        int
            Bucket index in ``[0, self.dim)``.
        """
        # Use abs() to handle negative hashes; mask to get positive int.
        # ``hash()`` is non-deterministic across Python invocations (random
        # seed), but that is fine — we only need consistency *within* a
        # single process run for search to work correctly.
        return hash(ngram) % self.dim

    def _compute_raw_vector(self, text: str) -> List[float]:
        """Compute the raw (un-normalised) embedding vector for *text*.

        Steps:
        1. Tokenise the text.
        2. Compute token frequencies (TF).
        3. For each token, extract character n-grams, hash each into a
           bucket, and accumulate using sub-linear TF scaling.
        4. Return the dense vector.

        Parameters
        ----------
        text:
            The input text.

        Returns
        -------
        list[float]
            Raw embedding vector of length ``self.dim``.
        """
        vec = [0.0] * self.dim

        tokens = self._tokenize(text)
        if not tokens:
            return vec

        # Token frequency (TF) counts.
        tf_counts: Counter = Counter(tokens)

        for token, tf in tf_counts.items():
            # Sub-linear TF scaling: 1 + log(tf)
            weight = _TF_LOG_SCALE + math.log(tf) if tf > 0 else 0.0

            ngrams = self._extract_ngrams(token)
            for ngram in ngrams:
                bucket = self._hash_ngram(ngram)
                vec[bucket] += weight

        return vec

    @staticmethod
    def _l2_normalise(vec: List[float]) -> List[float]:
        """L2-normalise a vector to unit length in-place and return it.

        If the vector has zero magnitude, it is returned unchanged
        (all zeros).

        Parameters
        ----------
        vec:
            The vector to normalise.

        Returns
        -------
        list[float]
            The same list, with values scaled so that
            ``sum(v*v for v in vec) == 1.0``.
        """
        mag_sq = sum(v * v for v in vec)
        if mag_sq == 0.0:
            return vec

        mag = math.sqrt(mag_sq)
        inv_mag = 1.0 / mag
        for i in range(len(vec)):
            vec[i] *= inv_mag

        return vec
