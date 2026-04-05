"""
Knowledge Search Engine.

Advanced search capabilities over the knowledge base — combining TF-IDF
semantic similarity, keyword matching, fuzzy search (Levenshtein distance),
and graph-based ranking from the knowledge graph.

All search strategies are composable through :meth:`KnowledgeSearch.search`,
which merges results with configurable weights.  A lightweight TF-IDF index
is maintained entirely in memory (no external libraries) and rebuilt on
demand via :meth:`rebuild_index`.

Usage::

    from agent.knowledge_base.knowledge_search import KnowledgeSearch, SearchQuery

    search_engine = KnowledgeSearch(store, graph=graph)
    await search_engine.initialize()

    results = await search_engine.search(SearchQuery(
        query="FastAPI dependency injection",
        categories=["pattern", "solution"],
        max_results=10,
        sort_by="relevance",
    ))

    # Auto-complete suggestions
    suggestions = await search_engine.suggest("fastapi dep", limit=5)

    # Format results for an LLM prompt
    context = await search_engine.get_context_for_prompt("how to handle auth")
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    from agent.knowledge_base.knowledge_graph import KnowledgeGraph
    from agent.knowledge_base.knowledge_store import KnowledgeEntry, KnowledgeStore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Tokenisation pattern — same as knowledge_store's ``_TOKEN_RE``.
_TOKEN_RE: re.Pattern = re.compile(r"[a-z0-9]+", re.IGNORECASE)

#: Stop words imported from the store module to stay in sync.  We duplicate
#: the set here to avoid a circular import at runtime.
_STOP_WORDS: FrozenSet[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren", "arent", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "couldn", "couldnt", "d", "did", "didn", "didnt", "do", "does",
    "doesn", "doesnt", "doing", "don", "dont", "down", "during", "each",
    "few", "for", "from", "further", "get", "got", "had", "hadn", "hadnt",
    "has", "hasn", "hasnt", "have", "haven", "havent", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "id", "if", "in", "into", "is", "isn", "isnt", "it", "its",
    "itself", "just", "ll", "let", "m", "ma", "me", "might", "mightn",
    "more", "most", "must", "mustn", "my", "myself", "need", "no", "nor",
    "not", "now", "o", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same",
    "shan", "she", "should", "shouldn", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "ve", "was", "wasn", "we", "were", "weren",
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "won", "would", "wouldn", "you", "youd", "your",
    "yours", "yourself", "yourselves",
})

#: Maximum character window for highlight extraction.
_HIGHLIGHT_WINDOW: int = 50

#: Padding string used when a highlight is truncated.
_HIGHLIGHT_PADDING: str = "..."

#: Minimum similarity score for fuzzy matches to be considered.
_FUZZY_MIN_SIMILARITY: float = 0.6

#: Default weighting for each search strategy in combined search.
_DEFAULT_WEIGHT_SEMANTIC: float = 0.4
_DEFAULT_WEIGHT_KEYWORD: float = 0.3
_DEFAULT_WEIGHT_FUZZY: float = 0.15
_DEFAULT_WEIGHT_GRAPH: float = 0.15

#: Number of characters per token for a rough ``max_tokens`` budget
#: when formatting context for prompts.
_CHARS_PER_TOKEN_APPROX: int = 4

#: Valid ``sort_by`` values accepted by :class:`SearchQuery`.
_VALID_SORT_KEYS: Tuple[str, ...] = (
    "relevance",
    "importance",
    "confidence",
    "recent",
    "accessed",
)

#: Valid ``match_type`` annotations for :class:`SearchResult`.
_VALID_MATCH_TYPES: Tuple[str, ...] = ("semantic", "keyword", "fuzzy", "graph")


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single search result returned by any search strategy.

    Attributes
    ----------
    entry:
        The matched :class:`KnowledgeEntry`.
    score:
        Relevance / similarity score in the range [0.0, 1.0].
    match_type:
        Which search strategy produced this result — one of
        ``"semantic"``, ``"keyword"``, ``"fuzzy"``, or ``"graph"``.
        In a combined search this is set to the highest-scoring strategy.
    highlights:
        Short snippets from the entry content that contain matched text,
        with ``...`` padding around each window.
    rank_position:
        Zero-based rank after sorting.  Populated after deduplication
        and final re-ranking.
    """

    entry: "KnowledgeEntry"
    score: float = 0.0
    match_type: str = "keyword"
    highlights: List[str] = field(default_factory=list)
    rank_position: int = 0


@dataclass
class SearchQuery:
    """Structured query object for advanced knowledge base search.

    Attributes
    ----------
    query:
        Free-text search string.
    categories:
        Optional list of categories to restrict results to.  When non-empty
        only entries whose ``category`` is in this list are returned.
    tags:
        Optional list of tags to filter by.  Entries must have **all**
        listed tags to appear in results.
    min_confidence:
        Minimum ``confidence`` threshold (default 0.0 — accept all).
    min_importance:
        Minimum ``importance`` threshold (default 0.0 — accept all).
    source_filter:
        Optional source filter — only return entries whose ``source``
        matches this string exactly.
    max_results:
        Maximum number of results to return (default 20).
    sort_by:
        Result ordering strategy.  Accepted values:

        - ``"relevance"``  — search score (default)
        - ``"importance"`` — entry importance field
        - ``"confidence"`` — entry confidence field
        - ``"recent"``     — ``created_at`` descending
        - ``"accessed"``   — ``access_count`` descending
    include_graph:
        When ``True``, graph-based results are blended into the combined
        score (requires a :class:`KnowledgeGraph` to be provided at init).
    search_depth:
        Depth for graph traversal when ``include_graph`` is ``True``
        (default 2).
    """

    query: str = ""
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    min_confidence: float = 0.0
    min_importance: float = 0.0
    source_filter: str = ""
    max_results: int = 20
    sort_by: str = "relevance"
    include_graph: bool = False
    search_depth: int = 2


# ──────────────────────────────────────────────────────────────────────────────
# Tokenisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric boundaries, drop stop-words.

    Mirrors :func:`agent.knowledge_base.knowledge_store._tokenize` so that
    both modules produce identical token streams.

    Parameters
    ----------
    text:
        Raw text to tokenise.

    Returns
    -------
    list[str]
        Cleaned, lowercased tokens longer than one character.
    """
    raw = _TOKEN_RE.findall(text.lower())
    return [tok for tok in raw if tok not in _STOP_WORDS and len(tok) > 1]


def _tokenize_keep_all(text: str) -> List[str]:
    """Tokenise without filtering stop-words (used for fuzzy matching).

    Parameters
    ----------
    text:
        Raw text to tokenise.

    Returns
    -------
    list[str]
        Lowercased tokens longer than one character, stop-words included.
    """
    raw = _TOKEN_RE.findall(text.lower())
    return [tok for tok in raw if len(tok) > 1]


# ──────────────────────────────────────────────────────────────────────────────
# Highlighting
# ──────────────────────────────────────────────────────────────────────────────

def _extract_highlights(
    text: str,
    match_tokens: Set[str],
    window: int = _HIGHLIGHT_WINDOW,
    max_highlights: int = 3,
) -> List[str]:
    """Extract context windows around matching tokens.

    For each matched token found in *text*, a window of up to *window*
    characters centred on the match is extracted.  Each snippet is padded
    with ``...`` on the left and/or right if the window does not reach
    the beginning or end of the text.

    Parameters
    ----------
    text:
        The full content string to search within.
    match_tokens:
        Set of lowercased tokens that qualify as matches.
    window:
        Half-width of the context window in characters (default 50).
        The total window is up to ``2 * window + len(token)`` characters.
    max_highlights:
        Maximum number of distinct highlights to return.

    Returns
    -------
    list[str]
        Highlight snippets, at most *max_highlights* in length.
        May be empty if no tokens match.
    """
    if not text or not match_tokens:
        return []

    text_lower = text.lower()
    highlights: List[str] = []
    seen_spans: List[Tuple[int, int]] = []

    for token in sorted(match_tokens, key=len, reverse=True):
        if len(highlights) >= max_highlights:
            break
        start: int = 0
        while True:
            idx = text_lower.find(token, start)
            if idx == -1:
                break
            # Ensure this span does not overlap with an existing one.
            span_end = idx + len(token)
            overlaps = any(
                not (span_end < s or idx > e) for s, e in seen_spans
            )
            if not overlaps:
                # Compute the context window.
                left = max(0, idx - window)
                right = min(len(text), span_end + window)
                snippet = ""
                if left > 0:
                    snippet += _HIGHLIGHT_PADDING
                snippet += text[left:right]
                if right < len(text):
                    snippet += _HIGHLIGHT_PADDING
                highlights.append(snippet)
                seen_spans.append((left, right))
                if len(highlights) >= max_highlights:
                    break
            start = idx + 1

    return highlights


# ──────────────────────────────────────────────────────────────────────────────
# Levenshtein distance
# ──────────────────────────────────────────────────────────────────────────────

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.

    Uses the standard dynamic-programming algorithm with O(min(m, n))
    space optimisation where *m* and *n* are the lengths of *s1* and *s2*.

    Parameters
    ----------
    s1:
        First string.
    s2:
        Second string.

    Returns
    -------
    int
        The minimum number of single-character insertions, deletions,
        and substitutions required to transform *s1* into *s2*.

    Examples
    --------
    >>> _levenshtein_distance("kitten", "sitting")
    3
    >>> _levenshtein_distance("", "abc")
    3
    >>> _levenshtein_distance("abc", "abc")
    0
    """
    # Always iterate over the shorter string for space efficiency.
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if not s2:
        return len(s1)

    previous_row: List[int] = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row: List[int] = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of substitution: 0 if characters match, 1 otherwise.
            cost = 0 if c1 == c2 else 1
            current_row.append(
                min(
                    current_row[j] + 1,          # insertion
                    previous_row[j + 1] + 1,      # deletion
                    previous_row[j] + cost,        # substitution
                )
            )
        previous_row = current_row

    return previous_row[-1]


def _fuzzy_similarity(s1: str, s2: str) -> float:
    """Return a similarity score in [0.0, 1.0] based on edit distance.

    The score is computed as::

        similarity = 1.0 - (distance / max_length)

    where ``max_length`` is ``max(len(s1), len(s2))``.  If both strings
    are empty the result is ``1.0``.

    Parameters
    ----------
    s1:
        First string.
    s2:
        Second string.

    Returns
    -------
    float
        Similarity score where 1.0 means identical and 0.0 means maximally
        different.
    """
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    distance = _levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF engine (in-memory, no external dependencies)
# ──────────────────────────────────────────────────────────────────────────────

class _TfIdfIndex:
    """Lightweight in-memory TF-IDF index over knowledge entries.

    The index maps each entry id to its TF-IDF vector (a sparse
    ``dict[str, float]`` mapping term → tf-idf weight).  IDF values are
    cached and reused across queries.

    Parameters
    ----------
    entries:
        List of ``(entry_id, title, content)`` tuples to index.
    """

    def __init__(
        self,
        entries: Optional[List[Tuple[str, str, str]]] = None,
    ) -> None:
        # Mapping: entry_id -> sparse TF-IDF vector {term: weight}
        self._vectors: Dict[str, Dict[str, float]] = {}
        # Cached IDF: term -> inverse-document-frequency
        self._idf: Dict[str, float] = {}
        # Vocabulary (all terms seen during indexing)
        self._vocabulary: Set[str] = set()
        # Total number of documents indexed
        self._n_docs: int = 0
        # Document frequency: term -> number of docs containing the term
        self._df: Dict[str, int] = {}

        if entries:
            self.build(entries)

    # ── Index construction ────────────────────────────────────────────────

    def build(self, entries: List[Tuple[str, str, str]]) -> None:
        """Build (or rebuild) the TF-IDF index from scratch.

        Parameters
        ----------
        entries:
            List of ``(entry_id, title, content)`` tuples.  Both title
            and content are tokenised and combined into a single bag-of-
            words representation per entry.
        """
        self._vectors.clear()
        self._idf.clear()
        self._vocabulary.clear()
        self._df.clear()

        if not entries:
            self._n_docs = 0
            logger.debug("TF-IDF index built with 0 entries")
            return

        self._n_docs = len(entries)

        # ── Pass 1: tokenise and compute document frequencies ────────
        tokenised_docs: Dict[str, List[str]] = {}
        for entry_id, title, content in entries:
            # Combine title (boosted) + content tokens.
            title_tokens = _tokenize(title)
            content_tokens = _tokenize(content)
            # Weight title tokens by repeating them to increase their
            # influence on TF (title appears ~3× effectively).
            combined = title_tokens * 3 + content_tokens
            tokenised_docs[entry_id] = combined

            # Track unique terms per document for DF.
            unique_terms = set(combined)
            for term in unique_terms:
                self._df[term] = self._df.get(term, 0) + 1

        self._vocabulary = set(self._df.keys())

        # ── Pass 2: compute IDF ──────────────────────────────────────
        for term, df in self._df.items():
            # Smooth IDF: log((N + 1) / (df + 1)) + 1  — prevents
            # division-by-zero and dampens the effect of rare terms.
            self._idf[term] = math.log((self._n_docs + 1) / (df + 1)) + 1.0

        # ── Pass 3: compute TF-IDF vectors per entry ─────────────────
        for entry_id, tokens in tokenised_docs.items():
            if not tokens:
                self._vectors[entry_id] = {}
                continue
            tf_counts: Counter = Counter(tokens)
            max_tf = max(tf_counts.values()) if tf_counts else 1

            vector: Dict[str, float] = {}
            for term, count in tf_counts.items():
                # Normalised TF: 0.5 + 0.5 * (count / max_tf)
                tf = 0.5 + 0.5 * (count / max_tf)
                idf = self._idf.get(term, 1.0)
                vector[term] = tf * idf

            self._vectors[entry_id] = vector

        logger.debug(
            "TF-IDF index built with %d entries, %d unique terms",
            self._n_docs,
            len(self._vocabulary),
        )

    # ── Querying ──────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        limit: int = 20,
    ) -> List[Tuple[str, float]]:
        """Score all entries against a query string using cosine similarity.

        Parameters
        ----------
        query_text:
            Free-text query to compare against indexed entries.
        limit:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            ``(entry_id, score)`` pairs sorted by descending score.
            Scores are in [0.0, 1.0].
        """
        if not self._vectors or not query_text.strip():
            return []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        # Build the query TF-IDF vector.
        tf_counts: Counter = Counter(query_tokens)
        max_tf = max(tf_counts.values()) if tf_counts else 1
        query_vector: Dict[str, float] = {}
        for term, count in tf_counts.items():
            tf = 0.5 + 0.5 * (count / max_tf)
            idf = self._idf.get(term, 1.0)
            query_vector[term] = tf * idf

        if not query_vector:
            return []

        # Pre-compute query vector magnitude for cosine similarity.
        query_magnitude = math.sqrt(
            sum(v * v for v in query_vector.values())
        )
        if query_magnitude == 0.0:
            return []

        # Score each entry.
        scores: List[Tuple[str, float]] = []
        for entry_id, doc_vector in self._vectors.items():
            # Dot product (only over terms present in both vectors).
            dot = 0.0
            shared_terms = set(query_vector) & set(doc_vector)
            for term in shared_terms:
                dot += query_vector[term] * doc_vector[term]

            if dot == 0.0:
                continue

            # Document vector magnitude.
            doc_magnitude = math.sqrt(
                sum(v * v for v in doc_vector.values())
            )
            if doc_magnitude == 0.0:
                continue

            # Cosine similarity in [0.0, 1.0].
            cosine = dot / (query_magnitude * doc_magnitude)
            scores.append((entry_id, round(min(cosine, 1.0), 6)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]

    @property
    def is_empty(self) -> bool:
        """Return ``True`` when the index contains no documents."""
        return self._n_docs == 0

    @property
    def n_docs(self) -> int:
        """Number of documents in the index."""
        return self._n_docs

    @property
    def vocabulary_size(self) -> int:
        """Number of unique terms in the vocabulary."""
        return len(self._vocabulary)


# ──────────────────────────────────────────────────────────────────────────────
# KnowledgeSearch
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeSearch:
    """Advanced search engine over the knowledge base.

    Combines four search strategies — TF-IDF semantic similarity, keyword
    matching, fuzzy search, and graph-based ranking — into a unified
    interface.  Each strategy can be used independently or blended via
    :meth:`combined_search` with configurable weights.

    The TF-IDF index is built from all entries in the store on
    :meth:`initialize` and can be rebuilt at any time with
    :meth:`rebuild_index`.

    Parameters
    ----------
    store:
        An initialised :class:`KnowledgeStore` instance.
    graph:
        Optional :class:`KnowledgeGraph` instance for graph-based
        search.  When ``None``, graph-related features are disabled.

    Examples
    --------
    >>> search = KnowledgeSearch(store, graph=graph)
    >>> await search.initialize()
    >>> results = await search.search(SearchQuery(query="python async"))
    """

    def __init__(
        self,
        store: "KnowledgeStore",
        graph: Optional["KnowledgeGraph"] = None,
    ) -> None:
        self._store = store
        self._graph = graph
        self._tfidf = _TfIdfIndex()
        # Cache of all entries (id -> KnowledgeEntry) for fast lookup
        # after a search returns entry ids.
        self._entries_cache: Dict[str, "KnowledgeEntry"] = {}
        # In-memory set of all entry tokens for auto-complete suggestions.
        self._all_tokens: Counter = Counter()
        self._all_titles: List[str] = []
        self._initialised: bool = False

    # ── Initialisation / lifecycle ─────────────────────────────────────────

    async def initialize(self) -> None:
        """Build the TF-IDF index and warm internal caches from the store.

        Must be called once after construction before issuing any search
        queries.  Subsequent calls rebuild the index from scratch.

        Raises
        ------
        RuntimeError
            If the underlying :class:`KnowledgeStore` has not been
            initialised (no database connection).
        """
        await self.rebuild_index()
        self._initialised = True
        logger.info("KnowledgeSearch engine initialized")

    async def rebuild_index(self) -> None:
        """Rebuild the in-memory TF-IDF index from the store's entries.

        Fetches all non-archived entries, tokenises their titles and
        content, and rebuilds the inverted index, IDF cache, and
        suggestion dictionary.

        This is an async wrapper that delegates the CPU-bound indexing
        work to the thread-pool executor.
        """
        entries = await self._store.search("", limit=100_000)
        if not entries:
            logger.debug("No entries found — building empty TF-IDF index")
            self._tfidf.build([])
            self._entries_cache.clear()
            self._all_tokens.clear()
            self._all_titles.clear()
            return

        # Build entry cache and raw data for TF-IDF.
        raw_data: List[Tuple[str, str, str]] = []
        token_counter: Counter = Counter()
        titles: List[str] = []

        for entry in entries:
            self._entries_cache[entry.id] = entry
            raw_data.append((entry.id, entry.title, entry.content))
            titles.append(entry.title)

            # Tokenise for the suggestion index.
            all_tokens = _tokenize(entry.title + " " + entry.content)
            token_counter.update(all_tokens)
            # Also index tag tokens for suggestions.
            for tag in entry.tags:
                tag_tokens = _tokenize(tag)
                token_counter.update(tag_tokens)

        self._all_tokens = token_counter
        self._all_titles = titles

        # Build TF-IDF index in the executor to avoid blocking the loop.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._tfidf.build, raw_data)

        logger.info(
            "TF-IDF index rebuilt: %d entries, %d terms",
            self._tfidf.n_docs,
            self._tfidf.vocabulary_size,
        )

    # ── Main search entry point ───────────────────────────────────────────

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute a structured search combining multiple strategies.

        This is the primary public API.  It:

        1. Delegates to :meth:`combined_search` for multi-strategy scoring.
        2. Applies category, tag, confidence, importance, and source filters.
        3. Sorts results according to ``query.sort_by``.
        4. Assigns final rank positions.
        5. Limits the result set to ``query.max_results``.

        Parameters
        ----------
        query:
            A fully-populated :class:`SearchQuery`.

        Returns
        -------
        list[SearchResult]
            Filtered, ranked, and limited search results.

        Raises
        ------
        ValueError
            If ``query.sort_by`` is not one of the valid sort keys.
        """
        # Validate sort key.
        if query.sort_by not in _VALID_SORT_KEYS:
            raise ValueError(
                f"Invalid sort_by={query.sort_by!r}. "
                f"Must be one of: {list(_VALID_SORT_KEYS)}"
            )

        if not query.query.strip():
            return []

        # Run combined search.
        results = await self.combined_search(query)

        # ── Post-filtering ────────────────────────────────────────────
        filtered: List[SearchResult] = []
        for result in results:
            entry = result.entry

            # Category filter.
            if query.categories and entry.category not in query.categories:
                continue

            # Tag filter (all tags must be present).
            if query.tags:
                entry_tag_set = set(entry.tags) if entry.tags else set()
                if not entry_tag_set.issuperset(set(query.tags)):
                    continue

            # Confidence threshold.
            if entry.confidence < query.min_confidence:
                continue

            # Importance threshold.
            if entry.importance < query.min_importance:
                continue

            # Source filter.
            if query.source_filter and entry.source != query.source_filter:
                continue

            filtered.append(result)

        # ── Sorting ───────────────────────────────────────────────────
        if query.sort_by == "relevance":
            filtered.sort(key=lambda r: r.score, reverse=True)
        elif query.sort_by == "importance":
            filtered.sort(key=lambda r: r.entry.importance, reverse=True)
        elif query.sort_by == "confidence":
            filtered.sort(key=lambda r: r.entry.confidence, reverse=True)
        elif query.sort_by == "recent":
            filtered.sort(key=lambda r: r.entry.created_at, reverse=True)
        elif query.sort_by == "accessed":
            filtered.sort(key=lambda r: r.entry.access_count, reverse=True)

        # ── Assign rank positions and limit ───────────────────────────
        for idx, result in enumerate(filtered[:query.max_results]):
            result.rank_position = idx

        return filtered[:query.max_results]

    # ── Individual search strategies ──────────────────────────────────────

    async def semantic_search(
        self,
        query: str,
        limit: int = 20,
    ) -> List[SearchResult]:
        """TF-IDF cosine similarity search over the knowledge base.

        Tokenises the query, computes a TF-IDF query vector, and
        calculates cosine similarity against every indexed entry.

        Parameters
        ----------
        query:
            Free-text search string.
        limit:
            Maximum number of results (default 20).

        Returns
        -------
        list[SearchResult]
            Results sorted by descending TF-IDF similarity score.
        """
        if not query.strip() or self._tfidf.is_empty:
            return []

        # Run TF-IDF query in executor (CPU-bound).
        loop = asyncio.get_running_loop()
        scored: List[Tuple[str, float]] = await loop.run_in_executor(
            None, self._tfidf.query, query, limit,
        )

        results: List[SearchResult] = []
        query_tokens = set(_tokenize(query))

        for entry_id, score in scored:
            entry = self._entries_cache.get(entry_id)
            if entry is None:
                continue
            highlights = _extract_highlights(entry.content, query_tokens)
            results.append(SearchResult(
                entry=entry,
                score=score,
                match_type="semantic",
                highlights=highlights,
                rank_position=0,
            ))

        return results

    async def keyword_search(
        self,
        query: str,
        limit: int = 20,
    ) -> List[SearchResult]:
        """Token-based keyword matching with scoring and highlighting.

        Tokenises the query and scores each entry based on:

        * **Title matches** — tokens appearing in the title (weight ×3)
        * **Content matches** — tokens appearing in the content (weight ×1)
        * **Tag matches** — query tokens matching entry tags (weight ×2)
        * **Importance bonus** — ``entry.importance * 2``
        * **Confidence bonus** — ``entry.confidence``

        Parameters
        ----------
        query:
            Free-text search string.
        limit:
            Maximum number of results (default 20).

        Returns
        -------
        list[SearchResult]
            Results sorted by descending keyword score, normalised to
            [0.0, 1.0].
        """
        if not query.strip():
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            # Fallback to raw substring matching.
            query_tokens = {query.strip().lower()}
        if not query_tokens:
            return []

        scored: List[Tuple[float, "KnowledgeEntry"]] = []

        for entry in self._entries_cache.values():
            title_tokens = set(_tokenize(entry.title))
            content_tokens = set(_tokenize(entry.content))
            entry_tags = set(t.lower() for t in entry.tags) if entry.tags else set()

            score = 0.0
            # Title matches are weighted higher.
            title_hits = len(title_tokens & query_tokens)
            score += title_hits * 3.0
            # Content matches.
            content_hits = len(content_tokens & query_tokens)
            score += content_hits * 1.0
            # Tag matches.
            tag_hits = len(entry_tags & query_tokens)
            score += tag_hits * 2.0
            # Importance and confidence bonuses.
            score += entry.importance * 2.0
            score += entry.confidence

            if score > 0:
                scored.append((score, entry))

        if not scored:
            return []

        # Normalise scores to [0.0, 1.0].
        max_score = max(s for s, _ in scored) if scored else 1.0
        if max_score == 0:
            max_score = 1.0

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResult] = []
        for raw_score, entry in scored[:limit]:
            normalised = round(raw_score / max_score, 6)
            highlights = _extract_highlights(entry.content, query_tokens)
            results.append(SearchResult(
                entry=entry,
                score=normalised,
                match_type="keyword",
                highlights=highlights,
                rank_position=0,
            ))

        return results

    async def fuzzy_search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Levenshtein-distance-based fuzzy matching.

        Tokenises the query and compares each query token against every
        token in every entry using :func:`_levenshtein_distance`.  For each
        entry the best token similarity is kept; the entry-level score is
        the average of the best similarities across all query tokens.

        Only entries with an average similarity above
        :data:`_FUZZY_MIN_SIMILARITY` (0.6) are returned.

        Parameters
        ----------
        query:
            Free-text search string.
        limit:
            Maximum number of results (default 10).

        Returns
        -------
        list[SearchResult]
            Results sorted by descending fuzzy similarity score.
        """
        if not query.strip():
            return []

        query_tokens = _tokenize_keep_all(query)
        if not query_tokens:
            return []

        # Pre-tokenise all entries (keep all tokens for fuzzy matching).
        entry_tokens_map: Dict[str, List[str]] = {}
        for entry_id, entry in self._entries_cache.items():
            entry_tokens_map[entry_id] = _tokenize_keep_all(
                entry.title + " " + entry.content
            )

        scored: List[Tuple[float, str, Set[str]]] = []

        for entry_id, entry_tokens in entry_tokens_map.items():
            best_similarities: List[float] = []
            matched_terms: Set[str] = set()

            for qt in query_tokens:
                best_sim = 0.0
                best_term = ""
                for et in entry_tokens:
                    sim = _fuzzy_similarity(qt, et)
                    if sim > best_sim:
                        best_sim = sim
                        best_term = et
                if best_sim >= _FUZZY_MIN_SIMILARITY:
                    best_similarities.append(best_sim)
                    if best_term:
                        matched_terms.add(best_term)

            if best_similarities:
                avg_sim = sum(best_similarities) / len(best_similarities)
                if avg_sim >= _FUZZY_MIN_SIMILARITY:
                    scored.append((avg_sim, entry_id, matched_terms))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResult] = []
        for score, entry_id, matched_terms in scored[:limit]:
            entry = self._entries_cache.get(entry_id)
            if entry is None:
                continue
            highlights = _extract_highlights(entry.content, matched_terms)
            results.append(SearchResult(
                entry=entry,
                score=round(score, 6),
                match_type="fuzzy",
                highlights=highlights,
                rank_position=0,
            ))

        return results

    async def graph_search(
        self,
        entry_id: str,
        depth: int = 2,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Traverse the knowledge graph from a given entry.

        Starting from *entry_id*, performs a BFS traversal up to *depth*
        hops through the knowledge graph.  Neighbouring entries are scored
        based on:

        * **Proximity**: closer entries receive higher scores.
        * **Edge weight**: stronger relationships score higher.
        * **Entry importance**: the neighbour's importance boosts its score.

        Parameters
        ----------
        entry_id:
            The id of the entry to start graph traversal from.
        depth:
            Maximum BFS depth (default 2).
        limit:
            Maximum number of results (default 10).

        Returns
        -------
        list[SearchResult]
            Graph-discovered entries sorted by descending proximity score.

        Raises
        ------
        RuntimeError
            If no :class:`KnowledgeGraph` was provided at construction.
        """
        if self._graph is None:
            logger.warning("Graph search requested but no KnowledgeGraph provided")
            return []

        try:
            neighbors = await self._graph.get_neighbors(entry_id, depth=depth)
        except Exception as exc:
            logger.warning("Graph traversal failed for entry %s: %s", entry_id, exc)
            return []

        # Flatten neighbours across all depth levels, keeping track of
        # the shallowest depth at which each node was discovered.
        node_depth: Dict[str, int] = {}
        for level_str, nodes in neighbors.items():
            level = int(level_str)
            for node in nodes:
                nid = node.entry_id
                if nid not in node_depth:
                    node_depth[nid] = level

        if not node_depth:
            return []

        # Score and build results.
        max_depth_val = max(node_depth.values()) if node_depth else 1
        if max_depth_val == 0:
            max_depth_val = 1

        scored: List[Tuple[float, str]] = []
        for nid, lvl in node_depth.items():
            entry = self._entries_cache.get(nid)
            if entry is None:
                # Try to fetch from the store.
                entry = await self._store.get(nid)
                if entry is None:
                    continue
                self._entries_cache[nid] = entry

            # Proximity score: closer → higher.
            proximity = 1.0 - (lvl / max_depth_val)
            # Importance boost.
            importance_bonus = entry.importance * 0.3
            # Confidence bonus.
            confidence_bonus = entry.confidence * 0.2

            combined = proximity + importance_bonus + confidence_bonus
            scored.append((combined, nid))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResult] = []
        for score, nid in scored[:limit]:
            entry = self._entries_cache.get(nid)
            if entry is None:
                continue
            # Highlight: mention the graph relationship.
            depth_label = node_depth.get(nid, 0)
            highlight = f"[Graph: {depth_label} hop{'s' if depth_label != 1 else ''} from source]"
            results.append(SearchResult(
                entry=entry,
                score=round(min(score, 1.0), 6),
                match_type="graph",
                highlights=[highlight],
                rank_position=0,
            ))

        return results

    # ── Combined search ───────────────────────────────────────────────────

    async def combined_search(
        self,
        query: SearchQuery,
    ) -> List[SearchResult]:
        """Merge results from all search strategies with weighted scoring.

        Each strategy is executed concurrently and the results are merged:

        * **Semantic** (TF-IDF):  weight 0.4
        * **Keyword**:            weight 0.3
        * **Fuzzy**:              weight 0.15
        * **Graph** (optional):   weight 0.15

        Entries appearing in multiple strategies receive a **boost** equal
        to ``0.05 * (n_strategies - 1)`` to reward consensus.  Results are
        deduplicated by ``entry_id``, keeping the entry with the highest
        combined score.

        If the graph is not available or ``query.include_graph`` is
        ``False``, the graph weight (0.15) is redistributed proportionally
        among the remaining strategies.

        Parameters
        ----------
        query:
            A :class:`SearchQuery` with at least ``query.query`` populated.

        Returns
        -------
        list[SearchResult]
            Deduplicated results sorted by descending combined score.
        """
        if not query.query.strip():
            return []

        # Determine weights — redistribute graph weight if graph is
        # unavailable or disabled.
        w_graph = _DEFAULT_WEIGHT_GRAPH if (query.include_graph and self._graph) else 0.0
        remaining = 1.0 - w_graph

        # Proportionally scale semantic / keyword / fuzzy weights.
        raw_total = (
            _DEFAULT_WEIGHT_SEMANTIC
            + _DEFAULT_WEIGHT_KEYWORD
            + _DEFAULT_WEIGHT_FUZZY
        )
        w_semantic = (_DEFAULT_WEIGHT_SEMANTIC / raw_total) * remaining
        w_keyword = (_DEFAULT_WEIGHT_KEYWORD / raw_total) * remaining
        w_fuzzy = (_DEFAULT_WEIGHT_FUZZY / raw_total) * remaining

        # Run strategies concurrently.
        semantic_results: List[SearchResult] = []
        keyword_results: List[SearchResult] = []
        fuzzy_results: List[SearchResult] = []
        graph_results: List[SearchResult] = []

        tasks = [
            self.semantic_search(query.query, limit=50),
            self.keyword_search(query.query, limit=50),
            self.fuzzy_search(query.query, limit=30),
        ]

        if w_graph > 0:
            # For graph search we need a seed entry.  We use the top
            # semantic/keyword result as the seed if available.
            # First run the other strategies, then seed the graph.
            pass

        # Execute the three core strategies in parallel.
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        if not isinstance(gathered[0], BaseException):
            semantic_results = gathered[0]
        if not isinstance(gathered[1], BaseException):
            keyword_results = gathered[1]
        if not isinstance(gathered[2], BaseException):
            fuzzy_results = gathered[2]

        # Run graph search if applicable.
        if w_graph > 0:
            # Use the top result from semantic/keyword as seed.
            seed_id: Optional[str] = None
            all_text_results = semantic_results + keyword_results
            if all_text_results:
                seed_id = all_text_results[0].entry.id

            if seed_id:
                try:
                    graph_results = await self.graph_search(
                        seed_id,
                        depth=query.search_depth,
                        limit=30,
                    )
                except Exception as exc:
                    logger.debug("Graph search skipped: %s", exc)

        # ── Merge scores ──────────────────────────────────────────────
        merged: Dict[str, Dict[str, Any]] = {}

        def _accumulate(
            results: List[SearchResult],
            weight: float,
            strategy_name: str,
        ) -> None:
            """Add weighted scores to the merged dict."""
            for r in results:
                eid = r.entry.id
                if eid not in merged:
                    merged[eid] = {
                        "entry": r.entry,
                        "weighted_score": 0.0,
                        "best_match_type": r.match_type,
                        "best_raw_score": r.score,
                        "strategy_count": 0,
                        "all_highlights": [],
                    }
                record = merged[eid]
                record["weighted_score"] += r.score * weight
                record["strategy_count"] += 1
                # Track the best match type (by raw score).
                if r.score > record["best_raw_score"]:
                    record["best_raw_score"] = r.score
                    record["best_match_type"] = r.match_type
                # Collect highlights, deduplicating.
                for h in r.highlights:
                    if h not in record["all_highlights"]:
                        record["all_highlights"].append(h)

        _accumulate(semantic_results, w_semantic, "semantic")
        _accumulate(keyword_results, w_keyword, "keyword")
        _accumulate(fuzzy_results, w_fuzzy, "fuzzy")
        if w_graph > 0:
            _accumulate(graph_results, w_graph, "graph")

        # ── Build final results with consensus boost ──────────────────
        final: List[SearchResult] = []
        for eid, record in merged.items():
            # Consensus boost: +0.05 per additional strategy.
            boost = 0.05 * max(0, record["strategy_count"] - 1)
            final_score = min(record["weighted_score"] + boost, 1.0)

            # Limit highlights to the most relevant ones (up to 5).
            highlights = record["all_highlights"][:5]

            final.append(SearchResult(
                entry=record["entry"],
                score=round(final_score, 6),
                match_type=record["best_match_type"],
                highlights=highlights,
                rank_position=0,
            ))

        final.sort(key=lambda r: r.score, reverse=True)
        return final

    # ── Auto-complete suggestions ─────────────────────────────────────────

    async def suggest(
        self,
        query: str,
        limit: int = 5,
    ) -> List[str]:
        """Provide auto-complete suggestions based on a query prefix.

        Suggestions are drawn from:

        1. **Entry titles** that start with the query prefix (case-insensitive).
        2. **Tokens** (individual words) that start with the query prefix,
           ranked by frequency in the knowledge base.

        Parameters
        ----------
        query:
            Partial query string (prefix to match).
        limit:
            Maximum number of suggestions (default 5).

        Returns
        -------
        list[str]
            Suggested completions, ordered by relevance.  Title completions
            are preferred over single-token completions.
        """
        if not query.strip():
            return []

        prefix = query.strip().lower()
        suggestions: List[Tuple[int, str]] = []

        # 1. Title-based suggestions (higher priority).
        seen_titles: Set[str] = set()
        for title in self._all_titles:
            if title.lower().startswith(prefix) and title not in seen_titles:
                suggestions.append((0, title))  # Priority 0 = highest.
                seen_titles.add(title)

        # 2. Token-based suggestions (sorted by frequency).
        token_suggestions: List[Tuple[int, str]] = []
        seen_tokens: Set[str] = set()
        for token, count in self._all_tokens.most_common():
            if len(token_suggestions) >= limit * 3:
                break
            if token.startswith(prefix) and token not in seen_tokens:
                token_suggestions.append((count, token))
                seen_tokens.add(token)

        # Sort token suggestions by descending frequency.
        token_suggestions.sort(key=lambda x: x[0], reverse=True)
        for freq, token in token_suggestions:
            if len(suggestions) >= limit:
                break
            suggestions.append((1, token))  # Priority 1 = second tier.

        # Sort by priority tier, then by the natural order within each tier.
        suggestions.sort(key=lambda x: x[0])
        return [s for _, s in suggestions[:limit]]

    # ── Context formatting for LLM prompts ───────────────────────────────

    async def get_context_for_prompt(
        self,
        query: str,
        max_tokens: int = 4000,
    ) -> str:
        """Format search results as a context block for LLM prompts.

        Runs a combined search and renders the top results into a compact
        Markdown-like format suitable for injection into a system or user
        prompt.  The output is truncated to approximately *max_tokens*
        tokens (using a rough 4-char-per-token heuristic).

        Each result is formatted as::

            ### [rank+1]. Title (category)
            Content excerpt...
            Tags: tag1, tag2

        Parameters
        ----------
        query:
            The search query string.
        max_tokens:
            Approximate maximum token budget for the output text
            (default 4000).

        Returns
        -------
        str
            Formatted context string, possibly empty if no results are
            found.  The string always starts with a ``<knowledge_context>``
            tag and ends with ``</knowledge_context>``.
    """
        search_query = SearchQuery(query=query, max_results=20, sort_by="relevance")
        results = await self.search(search_query)

        if not results:
            return ""

        # Rough character budget.
        max_chars = max_tokens * _CHARS_PER_TOKEN_APPROX
        parts: List[str] = ["<knowledge_context>"]
        current_len = len(parts[0]) + len("</knowledge_context>")

        for result in results:
            entry = result.entry

            # Build the entry block.
            header = f"### {result.rank_position + 1}. {entry.title} ({entry.category})"
            # Use highlights if available, otherwise truncate content.
            if result.highlights:
                excerpt = "\n".join(f"  > {h}" for h in result.highlights[:3])
            else:
                excerpt = entry.content[:300]
                if len(entry.content) > 300:
                    excerpt += "..."

            tag_line = ""
            if entry.tags:
                tag_line = f"Tags: {', '.join(entry.tags)}"

            block_parts = [header, excerpt]
            if tag_line:
                block_parts.append(tag_line)

            block = "\n".join(block_parts) + "\n"

            # Check if adding this block exceeds the budget.
            if current_len + len(block) > max_chars:
                break

            parts.append(block)
            current_len += len(block)

        parts.append("</knowledge_context>")
        return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "KnowledgeSearch",
    "SearchQuery",
    "SearchResult",
    "_TfIdfIndex",
    "_levenshtein_distance",
    "_fuzzy_similarity",
    "_extract_highlights",
    "_tokenize",
    "_tokenize_keep_all",
]
