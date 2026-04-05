"""
Codebase Indexer with semantic search for a Claude Code clone.

Provides full-project scanning, symbol extraction, reference graph construction,
TF-IDF based semantic search, incremental indexing, smart context selection
for LLM prompts, cross-file type inference, import resolution, and codebase
statistics — all backed by an in-memory cache and a persistent SQLite store.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".mjs": "javascript",
    ".cjs": "javascript",
}

DEFAULT_EXCLUDED_DIRS: list[str] = [
    ".git", ".hg", ".svn", "__pycache__", "node_modules", "venv", ".venv",
    "env", ".env", "dist", "build", ".build", "out", ".tox", ".mypy_cache",
    ".pytest_cache", ".eggs", "*.egg-info", ".idea", ".vscode", "coverage",
    ".nyc_output", ".next", ".nuxt", ".cache", "target", "vendor",
]

DEFAULT_EXCLUDED_FILES: set[str] = {
    ".DS_Store", "Thumbs.db", "package-lock.json", "yarn.lock",
    "pnpm-lock.yaml", "Cargo.lock", "go.sum", "composer.lock",
    "poetry.lock", "Pipfile.lock", "gems.locked",
}

STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are", "were",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not", "no",
    "this", "that", "these", "those", "if", "then", "else", "when", "while",
    "return", "def", "class", "func", "fn", "var", "let", "const", "import",
    "export", "from", "self", "cls", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "them", "its", "which", "what", "who", "how",
    "where", "why", "up", "down", "out", "just", "also", "than", "so",
    "very", "too", "only", "about", "into", "over", "after", "before",
})

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SymbolKind(Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    PROPERTY = "property"
    IMPORT = "import"
    EXPORT = "export"
    MODULE = "module"
    INTERFACE = "interface"
    TYPE = "type"
    ENUM = "enum"


class ReferenceKind(Enum):
    DEFINITION = "definition"
    CALL = "call"
    IMPORT = "import"
    ASSIGNMENT = "assignment"
    TYPE_ANNOTATION = "type_annotation"
    INHERITANCE = "inheritance"
    REFERENCE = "reference"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class Symbol:
    name: str
    kind: SymbolKind
    file_path: str
    line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    parent: str = ""
    module: str = ""
    language: str = ""
    hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reference:
    symbol_name: str
    file_path: str
    line: int
    kind: ReferenceKind
    context: str = ""


@dataclass
class FileIndex:
    path: str
    language: str
    symbols: list[Symbol] = field(default_factory=list)
    hash: str = ""
    last_indexed: float = 0.0
    line_count: int = 0


@dataclass
class SearchResult:
    symbol: Symbol
    score: float
    match_type: str = "semantic"
    context: str = ""
    highlights: list[str] = field(default_factory=list)


@dataclass
class IndexStats:
    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    symbols_extracted: int = 0
    references_built: int = 0
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Tokenizer / TF-IDF helpers
# ---------------------------------------------------------------------------

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NONALNUM_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Split text on whitespace / punctuation, split camelCase and
    snake_case identifiers, lowercase, drop stop words and short tokens."""
    tokens: list[str] = []
    # Split on non-alphanumeric to get raw words
    raw = _NONALNUM_RE.split(text.lower())
    for word in raw:
        # split camelCase / PascalCase
        parts = _CAMEL_RE.split(word) if word else []
        # also split snake_case
        for part in parts:
            for sub in part.split("_"):
                sub = sub.strip()
                if len(sub) >= 2 and sub not in STOP_WORDS:
                    tokens.append(sub)
    return tokens


# ---------------------------------------------------------------------------
# Codebase Indexer
# ---------------------------------------------------------------------------


class CodebaseIndexer:
    """Scan, index, and enable fast semantic search over a codebase."""

    def __init__(
        self,
        project_path: str,
        db_path: str | None = None,
        excluded_dirs: list[str] | None = None,
    ) -> None:
        self.project_path = os.path.abspath(project_path)
        self.db_path = db_path or os.path.join(self.project_path, ".claude_index.db")
        self.excluded_dirs: set[str] = set(excluded_dirs or DEFAULT_EXCLUDED_DIRS)
        self.excluded_files: set[str] = DEFAULT_EXCLUDED_FILES

        # In-memory caches
        self._symbols: dict[str, Symbol] = {}          # key: f"{file}:{name}:{line}"
        self._symbols_by_name: dict[str, list[Symbol]] = defaultdict(list)
        self._symbols_by_file: dict[str, list[Symbol]] = defaultdict(list)
        self._symbols_by_kind: dict[SymbolKind, list[Symbol]] = defaultdict(list)
        self._file_hashes: dict[str, str] = {}
        self._file_indices: dict[str, FileIndex] = {}
        self._references: dict[str, list[Reference]] = defaultdict(list)
        self._reverse_refs: dict[str, list[Reference]] = defaultdict(list)
        self._callers: dict[str, list[Symbol]] = defaultdict(list)
        self._callees: dict[str, list[Symbol]] = defaultdict(list)

        # TF-IDF state
        self._vocabulary: dict[str, int] = {}          # term -> index
        self._idf: dict[str, float] = {}
        self._tfidf_vectors: dict[str, list[float]] = {}  # symbol key -> vector
        self._vocab_dirty = False

        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization / DB
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create SQLite tables and load any existing index."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                language TEXT,
                file_hash TEXT,
                last_indexed REAL,
                line_count INTEGER
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                kind TEXT,
                file_path TEXT,
                line INTEGER,
                end_line INTEGER,
                signature TEXT,
                docstring TEXT,
                parent TEXT,
                module TEXT,
                language TEXT,
                symbol_hash TEXT,
                metadata TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS references_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_name TEXT,
                file_path TEXT,
                line INTEGER,
                kind TEXT,
                context TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                symbol_id INTEGER PRIMARY KEY,
                symbol_key TEXT,
                vector TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tfidf_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_symbol ON references_table(symbol_name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_file ON references_table(file_path)")
        self._conn.commit()
        await self._load_existing_index()
        self._initialized = True

    async def _load_existing_index(self) -> None:
        """Load symbols, references and TF-IDF state from SQLite."""
        assert self._conn is not None
        cur = self._conn.cursor()

        # Files
        cur.execute("SELECT path, language, file_hash, last_indexed, line_count FROM files")
        for row in cur:
            self._file_hashes[row[0]] = row[2]
            self._file_indices[row[0]] = FileIndex(
                path=row[0], language=row[1], hash=row[2],
                last_indexed=row[3], line_count=row[4],
            )

        # Symbols
        cur.execute(
            "SELECT name, kind, file_path, line, end_line, signature, "
            "docstring, parent, module, language, symbol_hash, metadata FROM symbols"
        )
        for row in cur:
            meta = {}
            if row[11]:
                try:
                    meta = json.loads(row[11])
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            sym = Symbol(
                name=row[0], kind=SymbolKind(row[1]), file_path=row[2],
                line=row[3], end_line=row[4], signature=row[5] or "",
                docstring=row[6] or "", parent=row[7] or "", module=row[8] or "",
                language=row[9] or "", hash=row[10] or "", metadata=meta,
            )
            key = self._sym_key(sym)
            self._symbols[key] = sym
            self._symbols_by_name[sym.name].append(sym)
            self._symbols_by_file[sym.file_path].append(sym)
            self._symbols_by_kind[sym.kind].append(sym)

        # References
        cur.execute(
            "SELECT symbol_name, file_path, line, kind, context FROM references_table"
        )
        for row in cur:
            ref = Reference(
                symbol_name=row[0], file_path=row[1], line=row[2],
                kind=ReferenceKind(row[3]), context=row[4] or "",
            )
            self._references[ref.symbol_name].append(ref)
            self._reverse_refs[f"{ref.file_path}:{ref.line}"].append(ref)

        # Rebuild callers / callees from references
        for sym_key, refs in self._reverse_refs.items():
            for ref in refs:
                if ref.kind == ReferenceKind.CALL:
                    # sym_key's owner calls ref.symbol_name
                    caller_syms = self._resolve_symbols_at(ref.file_path, ref.line)
                    callee_syms = self._symbols_by_name.get(ref.symbol_name, [])
                    for cs in caller_syms:
                        for clee in callee_syms:
                            self._callers[ref.symbol_name].append(cs)
                            self._callees[self._sym_key(cs)].append(clee)

        # TF-IDF state
        cur.execute("SELECT key, value FROM tfidf_state")
        tfidf_raw: dict[str, str] = {}
        for row in cur:
            tfidf_raw[row[0]] = row[1]
        if "vocabulary" in tfidf_raw:
            self._vocabulary = json.loads(tfidf_raw["vocabulary"])
        if "idf" in tfidf_raw:
            self._idf = json.loads(tfidf_raw["idf"])

        # Embeddings
        cur.execute("SELECT symbol_key, vector FROM embeddings")
        for row in cur:
            key = row[0]
            vec = json.loads(row[1]) if row[1] else []
            self._tfidf_vectors[key] = vec

    def _sym_key(self, sym: Symbol) -> str:
        return f"{sym.file_path}:{sym.name}:{sym.line}"

    def _resolve_symbols_at(self, file_path: str, line: int) -> list[Symbol]:
        """Return symbols that *contain* the given line in *file_path*."""
        result: list[Symbol] = []
        for sym in self._symbols_by_file.get(file_path, []):
            if sym.line <= line <= sym.end_line:
                result.append(sym)
        return result

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_project(self) -> IndexStats:
        """Full project index — walk every supported file."""
        start = time.time()
        stats = IndexStats()
        existing_paths = set(self._file_hashes.keys())

        for root, dirs, files in os.walk(self.project_path):
            # Prune excluded dirs
            dirs[:] = [
                d for d in dirs
                if d not in self.excluded_dirs and not d.startswith(".")
                and not any(fnmatch_simple(d, pat) for pat in self.excluded_dirs if "*" in pat)
            ]
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, self.project_path)
                if fname in self.excluded_files:
                    continue
                if not self._is_supported_file(fpath):
                    continue
                existing_paths.discard(fpath)
                content_hash = self._file_content_hash(fpath)
                if content_hash == self._file_hashes.get(fpath):
                    stats.files_skipped += 1
                    continue
                fi = await self.index_file(fpath)
                if fi:
                    stats.files_indexed += 1
                    stats.symbols_extracted += len(fi.symbols)

        # Remove files that no longer exist
        for stale_path in existing_paths:
            await self.remove_file(stale_path)
            stats.files_removed += 1

        stats.duration_seconds = time.time() - start
        await self._build_references()
        await self._rebuild_tfidf()
        await self._persist()
        return stats

    async def update_index(self) -> IndexStats:
        """Incremental update — only re-index changed files."""
        start = time.time()
        stats = IndexStats()
        existing_paths = set(self._file_hashes.keys())

        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [
                d for d in dirs
                if d not in self.excluded_dirs and not d.startswith(".")
            ]
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                if fname in self.excluded_files:
                    continue
                if not self._is_supported_file(fpath):
                    continue
                existing_paths.discard(fpath)
                content_hash = self._file_content_hash(fpath)
                if content_hash == self._file_hashes.get(fpath):
                    stats.files_skipped += 1
                    continue
                fi = await self.index_file(fpath)
                if fi:
                    stats.files_indexed += 1
                    stats.symbols_extracted += len(fi.symbols)

        for stale_path in existing_paths:
            await self.remove_file(stale_path)
            stats.files_removed += 1

        stats.duration_seconds = time.time() - start
        await self._build_references()
        await self._rebuild_tfidf()
        await self._persist()
        return stats

    async def index_file(self, filepath: str) -> FileIndex | None:
        """Index a single file and return its FileIndex."""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            return None

        language = self._language_from_extension(filepath)
        if not language:
            return None

        # Remove previous data for this file
        await self._remove_file_symbols(filepath)

        file_hash = self._content_string_hash(content)
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

        symbols = self._extract_symbols(filepath, content, language)

        fi = FileIndex(
            path=filepath, language=language, symbols=symbols,
            hash=file_hash, last_indexed=time.time(), line_count=line_count,
        )

        # Update caches
        self._file_hashes[filepath] = file_hash
        self._file_indices[filepath] = fi
        for sym in symbols:
            key = self._sym_key(sym)
            self._symbols[key] = sym
            self._symbols_by_name[sym.name].append(sym)
            self._symbols_by_file[filepath].append(sym)
            self._symbols_by_kind[sym.kind].append(sym)

        return fi

    async def remove_file(self, filepath: str) -> None:
        """Remove a file from the index."""
        await self._remove_file_symbols(filepath)
        self._file_hashes.pop(filepath, None)
        self._file_indices.pop(filepath, None)
        if self._conn:
            self._conn.execute("DELETE FROM files WHERE path = ?", (filepath,))

    async def _remove_file_symbols(self, filepath: str) -> None:
        old_syms = self._symbols_by_file.pop(filepath, [])
        for sym in old_syms:
            key = self._sym_key(sym)
            self._symbols.pop(key, None)
            name_list = self._symbols_by_name.get(sym.name)
            if name_list:
                self._symbols_by_name[sym.name] = [s for s in name_list if s.file_path != filepath]
                if not self._symbols_by_name[sym.name]:
                    del self._symbols_by_name[sym.name]
            kind_list = self._symbols_by_kind.get(sym.kind)
            if kind_list:
                self._symbols_by_kind[sym.kind] = [s for s in kind_list if s.file_path != filepath]
            self._tfidf_vectors.pop(key, None)

        if self._conn:
            self._conn.execute("DELETE FROM symbols WHERE file_path = ?", (filepath,))
            self._conn.execute(
                "DELETE FROM embeddings WHERE symbol_key LIKE ?",
                (f"{filepath}:%",),
            )
            self._conn.execute("DELETE FROM references_table WHERE file_path = ?", (filepath,))

        # Clean references mentioning removed symbols
        removed_names = {s.name for s in old_syms}
        for name in removed_names:
            self._references[name] = [
                r for r in self._references.get(name, []) if r.file_path != filepath
            ]
            self._callers[name] = [
                s for s in self._callers.get(name, []) if s.file_path != filepath
            ]
            self._callees = {
                k: [s for s in v if s.file_path != filepath]
                for k, v in self._callees.items()
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist(self) -> None:
        if not self._conn:
            return
        conn = self._conn
        for fpath, fi in self._file_indices.items():
            conn.execute(
                "INSERT OR REPLACE INTO files (path, language, file_hash, last_indexed, line_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (fi.path, fi.language, fi.hash, fi.last_indexed, fi.line_count),
            )
        for key, sym in self._symbols.items():
            meta_str = json.dumps(sym.metadata) if sym.metadata else ""
            conn.execute(
                "INSERT OR REPLACE INTO symbols "
                "(name, kind, file_path, line, end_line, signature, docstring, "
                "parent, module, language, symbol_hash, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sym.name, sym.kind.value, sym.file_path, sym.line, sym.end_line,
                 sym.signature, sym.docstring, sym.parent, sym.module,
                 sym.language, sym.hash, meta_str),
            )
        # References — bulk replace
        conn.execute("DELETE FROM references_table")
        for name, refs in self._references.items():
            for ref in refs:
                conn.execute(
                    "INSERT INTO references_table (symbol_name, file_path, line, kind, context) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ref.symbol_name, ref.file_path, ref.line, ref.kind.value, ref.context),
                )
        # Embeddings
        conn.execute("DELETE FROM embeddings")
        for skey, vec in self._tfidf_vectors.items():
            conn.execute(
                "INSERT INTO embeddings (symbol_key, vector) VALUES (?, ?)",
                (skey, json.dumps(vec)),
            )
        # TF-IDF state
        conn.execute(
            "INSERT OR REPLACE INTO tfidf_state (key, value) VALUES (?, ?)",
            ("vocabulary", json.dumps(self._vocabulary)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO tfidf_state (key, value) VALUES (?, ?)",
            ("idf", json.dumps(self._idf)),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Symbol extraction
    # ------------------------------------------------------------------

    def _extract_symbols(self, filepath: str, content: str, language: str) -> list[Symbol]:
        extractors = {
            "python": self._extract_symbols_python,
            "javascript": self._extract_symbols_javascript,
            "typescript": self._extract_symbols_javascript,
        }
        extractor = extractors.get(language)
        if extractor:
            return extractor(filepath, content)
        return self._extract_symbols_generic(filepath, content, language)

    # ---- Python (AST-based) ----

    def _extract_symbols_python(self, filepath: str, content: str) -> list[Symbol]:
        symbols: list[Symbol] = []
        module_name = self._module_name(filepath)
        try:
            tree = ast.parse(content, filename=filepath)
        except SyntaxError:
            return symbols

        lines = content.splitlines()

        def _get_docstring(node: ast.AST) -> str:
            ds = ast.get_docstring(node)
            return ds or ""

        def _signature(node: ast.AST) -> str:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args_parts: list[str] = []
                for a in node.args.args:
                    part = a.arg
                    if a.annotation:
                        part += f": {ast.unparse(a.annotation)}"
                    args_parts.append(part)
                if node.args.vararg:
                    args_parts.append(f"*{node.args.vararg.arg}")
                if node.args.kwarg:
                    args_parts.append(f"**{node.args.kwarg.arg}")
                ret = ""
                if node.returns:
                    ret = f" -> {ast.unparse(node.returns)}"
                return f"def {node.name}({', '.join(args_parts)}){ret}"
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
                return f"class {node.name}({bases})"
            return ""

        def _end_line(node: ast.AST) -> int:
            if hasattr(node, "end_lineno") and node.end_lineno:
                return node.end_lineno
            return node.lineno

        def _visit(node: ast.AST, parent: str = "") -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_method = parent != ""
                kind = SymbolKind.METHOD if is_method else SymbolKind.FUNCTION
                decorators = []
                for dec in node.decorator_list:
                    try:
                        decorators.append(ast.unparse(dec))
                    except Exception:
                        decorators.append("?")
                sig = _signature(node)
                ds = _get_docstring(node)
                meta: dict[str, Any] = {}
                if decorators:
                    meta["decorators"] = decorators
                if isinstance(node, ast.AsyncFunctionDef):
                    meta["async"] = True
                # Extract type annotations info
                if node.returns:
                    try:
                        meta["return_type"] = ast.unparse(node.returns)
                    except Exception:
                        pass
                sym = Symbol(
                    name=node.name, kind=kind, file_path=filepath,
                    line=node.lineno, end_line=_end_line(node),
                    signature=sig, docstring=ds, parent=parent,
                    module=module_name, language="python",
                    hash=self._content_string_hash(sig),
                    metadata=meta,
                )
                symbols.append(sym)
                # Walk body for nested defs / classes
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        _visit(child, parent=node.name)

            elif isinstance(node, ast.ClassDef):
                sig = _signature(node)
                ds = _get_docstring(node)
                bases = []
                for b in node.bases:
                    try:
                        bases.append(ast.unparse(b))
                    except Exception:
                        bases.append("?")
                meta = {}
                if bases:
                    meta["bases"] = bases
                sym = Symbol(
                    name=node.name, kind=SymbolKind.CLASS, file_path=filepath,
                    line=node.lineno, end_line=_end_line(node),
                    signature=sig, docstring=ds, parent=parent,
                    module=module_name, language="python",
                    hash=self._content_string_hash(sig), metadata=meta,
                )
                symbols.append(sym)
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        _visit(child, parent=node.name)

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        is_const = target.id.isupper()
                        kind = SymbolKind.CONSTANT if is_const else SymbolKind.VARIABLE
                        val = ""
                        try:
                            val = ast.unparse(node.value)
                        except Exception:
                            val = "<complex>"
                        sig = f"{target.id} = {val}"
                        sym = Symbol(
                            name=target.id, kind=kind, file_path=filepath,
                            line=node.lineno, end_line=node.end_lineno or node.lineno,
                            signature=sig, parent=parent,
                            module=module_name, language="python",
                            hash=self._content_string_hash(sig),
                        )
                        symbols.append(sym)

            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                kind = SymbolKind.CONSTANT if node.target.id.isupper() else SymbolKind.VARIABLE
                ann = ""
                try:
                    ann = ast.unparse(node.annotation)
                except Exception:
                    ann = "?"
                val = ""
                if node.value:
                    try:
                        val = ast.unparse(node.value)
                    except Exception:
                        val = "<complex>"
                sig = f"{node.target.id}: {ann}"
                if val:
                    sig += f" = {val}"
                meta = {"type_annotation": ann}
                sym = Symbol(
                    name=node.target.id, kind=kind, file_path=filepath,
                    line=node.lineno, end_line=node.end_lineno or node.lineno,
                    signature=sig, parent=parent,
                    module=module_name, language="python",
                    hash=self._content_string_hash(sig), metadata=meta,
                )
                symbols.append(sym)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    sym = Symbol(
                        name=name, kind=SymbolKind.IMPORT, file_path=filepath,
                        line=node.lineno, end_line=node.end_lineno or node.lineno,
                        signature=f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""),
                        parent=parent, module=module_name, language="python",
                        metadata={"import_path": alias.name},
                    )
                    symbols.append(sym)

            elif isinstance(node, ast.ImportFrom):
                module_imported = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    sig = f"from {module_imported} import {alias.name}"
                    if alias.asname:
                        sig += f" as {alias.asname}"
                    sym = Symbol(
                        name=name, kind=SymbolKind.IMPORT, file_path=filepath,
                        line=node.lineno, end_line=node.end_lineno or node.lineno,
                        signature=sig, parent=parent, module=module_name,
                        language="python",
                        metadata={"import_module": module_imported, "import_name": alias.name},
                    )
                    symbols.append(sym)

        for node in ast.iter_child_nodes(tree):
            _visit(node)

        # Top-level module docstring
        ds = _get_docstring(tree)
        if ds:
            mod_sym = Symbol(
                name=module_name or os.path.basename(filepath), kind=SymbolKind.MODULE,
                file_path=filepath, line=1, end_line=1, docstring=ds,
                module=module_name, language="python",
            )
            symbols.insert(0, mod_sym)

        return symbols

    # ---- JavaScript / TypeScript (regex-based) ----

    _JS_PATTERNS: list[tuple[re.Pattern[str], SymbolKind, str]] = []  # populated below

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def _build_js_patterns(self) -> None:
        if self._JS_PATTERNS:
            return
        pats = [
            (re.compile(
                r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"
                r"(?:\s*:\s*([^{]+))?",
                re.MULTILINE,
            ), SymbolKind.FUNCTION, "function"),
            (re.compile(
                r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?(\w+)\s*=\s*(?:async\s+)?"
                r"\(([^)]*)\)\s*(?::\s*([^{]+))?\s*=>",
                re.MULTILINE,
            ), SymbolKind.FUNCTION, "arrow"),
            (re.compile(
                r"^(?:export\s+(?:default\s+)?)?class\s+(\w+)(?:\s+extends\s+(\w+))?",
                re.MULTILINE,
            ), SymbolKind.CLASS, "class"),
            (re.compile(
                r"(?:export\s+)?(?:interface|type)\s+(\w+)",
                re.MULTILINE,
            ), SymbolKind.INTERFACE, "interface"),
            (re.compile(
                r"^(?:export\s+)?(?:type|enum)\s+(\w+)",
                re.MULTILINE,
            ), SymbolKind.ENUM, "enum"),
            (re.compile(
                r"^(?:export\s+(?:default\s+)?)?const\s+(\w+)\s*(?::\s*([^=]+))?\s*=",
                re.MULTILINE,
            ), SymbolKind.CONSTANT, "const"),
            (re.compile(
                r"^(?:export\s+(?:default\s+)?)?let\s+(\w+)\s*(?::\s*([^=]+))?\s*=",
                re.MULTILINE,
            ), SymbolKind.VARIABLE, "let"),
            (re.compile(
                r"^(?:export\s+(?:default\s+)?)?var\s+(\w+)\s*(?::\s*([^=]+))?\s*=",
                re.MULTILINE,
            ), SymbolKind.VARIABLE, "var"),
            (re.compile(
                r"import\s+(?:\{([^}]+)\}|(\w+))\s+from\s+['\"]([^'\"]+)['\"]",
                re.MULTILINE,
            ), SymbolKind.IMPORT, "import"),
            (re.compile(
                r"import\s+(['\"][^'\"]+['\"])",
                re.MULTILINE,
            ), SymbolKind.IMPORT, "import_default"),
            (re.compile(
                r"export\s+(?:default\s+)?(\w+)",
                re.MULTILINE,
            ), SymbolKind.EXPORT, "export"),
        ]
        type(_)._JS_PATTERNS = pats  # type: ignore[attr-defined]

    def _extract_symbols_javascript(self, filepath: str, content: str) -> list[Symbol]:
        self._build_js_patterns()
        symbols: list[Symbol] = []
        module_name = self._module_name(filepath)
        lines = content.splitlines()
        language = "typescript" if filepath.endswith((".ts", ".tsx")) else "javascript"

        # Method definitions inside classes
        method_re = re.compile(
            r"^\s+(?:(?:async|static|private|protected|public|readonly|abstract)\s+)*"
            r"(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*\{",
            re.MULTILINE,
        )
        for m in method_re.finditer(content):
            name = m.group(1)
            if name in ("if", "for", "while", "switch", "catch", "constructor", "function"):
                continue
            ln = content[:m.start()].count("\n") + 1
            params = m.group(2).strip()
            ret = m.group(3).strip() if m.group(3) else ""
            sig = f"{name}({params})"
            if ret:
                sig += f": {ret}"
            sym = Symbol(
                name=name, kind=SymbolKind.METHOD, file_path=filepath,
                line=ln, end_line=ln,
                signature=sig, parent="", module=module_name, language=language,
                hash=self._content_string_hash(sig),
            )
            symbols.append(sym)

        for pattern, kind, pat_type in type(self)._JS_PATTERNS:  # type: ignore[attr-defined]
            for m in pattern.finditer(content):
                ln = content[:m.start()].count("\n") + 1
                if pat_type == "function":
                    name = m.group(1)
                    params = m.group(2).strip()
                    ret = m.group(3).strip() if m.group(3) else ""
                    sig = f"function {name}({params})"
                    if ret:
                        sig += f": {ret}"
                    ds = self._extract_jsdoc(lines, ln - 1)
                    is_export = content[:m.start()].rstrip().endswith("export") or \
                                "export" in content[max(0, m.start() - 10):m.start()]
                    meta: dict[str, Any] = {"is_export": is_export}
                    sym = Symbol(
                        name=name, kind=kind, file_path=filepath,
                        line=ln, end_line=ln,
                        signature=sig, docstring=ds, parent="",
                        module=module_name, language=language,
                        hash=self._content_string_hash(sig), metadata=meta,
                    )
                    symbols.append(sym)

                elif pat_type == "arrow":
                    name = m.group(1)
                    params = m.group(2).strip()
                    ret = m.group(3).strip() if m.group(3) else ""
                    sig = f"{name} = ({params}) =>"
                    if ret:
                        sig = f"{name}: ({params}) => {ret}"
                    ds = self._extract_jsdoc(lines, ln - 1)
                    sym = Symbol(
                        name=name, kind=kind, file_path=filepath,
                        line=ln, end_line=ln,
                        signature=sig, docstring=ds, parent="",
                        module=module_name, language=language,
                        hash=self._content_string_hash(sig),
                    )
                    symbols.append(sym)

                elif pat_type == "class":
                    name = m.group(1)
                    base = m.group(2) or ""
                    sig = f"class {name}" + (f" extends {base}" if base else "")
                    ds = self._extract_jsdoc(lines, ln - 1)
                    meta = {}
                    if base:
                        meta["extends"] = base
                    sym = Symbol(
                        name=name, kind=kind, file_path=filepath,
                        line=ln, end_line=ln,
                        signature=sig, docstring=ds, parent="",
                        module=module_name, language=language,
                        hash=self._content_string_hash(sig), metadata=meta,
                    )
                    symbols.append(sym)

                elif pat_type in ("interface", "enum"):
                    name = m.group(1)
                    sig = f"{pat_type} {name}"
                    sym = Symbol(
                        name=name, kind=kind, file_path=filepath,
                        line=ln, end_line=ln,
                        signature=sig, parent="",
                        module=module_name, language=language,
                        hash=self._content_string_hash(sig),
                    )
                    symbols.append(sym)

                elif pat_type in ("const", "let", "var"):
                    name = m.group(1)
                    type_ann = m.group(2).strip() if m.group(2) else ""
                    sig = f"{pat_type} {name}"
                    if type_ann:
                        sig += f": {type_ann}"
                    sym = Symbol(
                        name=name, kind=kind, file_path=filepath,
                        line=ln, end_line=ln,
                        signature=sig, parent="",
                        module=module_name, language=language,
                        hash=self._content_string_hash(sig),
                        metadata={"type_annotation": type_ann} if type_ann else {},
                    )
                    symbols.append(sym)

                elif pat_type == "import":
                    named = m.group(1)
                    default = m.group(2)
                    source = m.group(3)
                    if named:
                        for imp_name in named.split(","):
                            imp_name = imp_name.strip().split(" as ")
                            actual = imp_name[0].strip()
                            alias = imp_name[1].strip() if len(imp_name) > 1 else actual
                            sig = f"import {{ {actual}"
                            if alias != actual:
                                sig += f" as {alias}"
                            sig += f" }} from '{source}'"
                            sym = Symbol(
                                name=alias, kind=SymbolKind.IMPORT, file_path=filepath,
                                line=ln, end_line=ln, signature=sig, parent="",
                                module=module_name, language=language,
                                metadata={"import_module": source, "import_name": actual},
                            )
                            symbols.append(sym)
                    elif default:
                        sig = f"import {default} from '{source}'"
                        sym = Symbol(
                            name=default, kind=SymbolKind.IMPORT, file_path=filepath,
                            line=ln, end_line=ln, signature=sig, parent="",
                            module=module_name, language=language,
                            metadata={"import_module": source, "import_name": default},
                        )
                        symbols.append(sym)

                elif pat_type == "import_default":
                    source = m.group(1)
                    sig = f"import {source}"
                    sym = Symbol(
                        name=source.strip("\"'"), kind=SymbolKind.IMPORT,
                        file_path=filepath, line=ln, end_line=ln,
                        signature=sig, parent="", module=module_name, language=language,
                        metadata={"import_module": source.strip("\"'")},
                    )
                    symbols.append(sym)

                elif pat_type == "export":
                    name = m.group(1)
                    if name in ("function", "class", "const", "let", "var", "default", "from"):
                        continue
                    sym = Symbol(
                        name=name, kind=SymbolKind.EXPORT, file_path=filepath,
                        line=ln, end_line=ln,
                        signature=f"export {name}", parent="",
                        module=module_name, language=language,
                    )
                    symbols.append(sym)

        return symbols

    def _extract_jsdoc(self, lines: list[str], line_idx: int) -> str:
        """Extract JSDoc / TSDoc comment above the given line index."""
        if line_idx <= 0:
            return ""
        doc_lines: list[str] = []
        i = line_idx - 1
        while i >= 0:
            stripped = lines[i].strip()
            if stripped.startswith("/**") or stripped.startswith("*"):
                if stripped.startswith("/**"):
                    doc_lines.insert(0, stripped[3:].rstrip("*/").strip())
                elif stripped.startswith("* "):
                    doc_lines.insert(0, stripped[2:].strip())
                elif stripped == "*/":
                    pass
                elif stripped == "*":
                    doc_lines.insert(0, "")
                else:
                    doc_lines.insert(0, stripped.lstrip("*").strip())
                i -= 1
            elif stripped.startswith("//"):
                doc_lines.insert(0, stripped[2:].strip())
                i -= 1
            else:
                break
        return "\n".join(doc_lines).strip()

    # ---- Generic (regex-based for Go, Rust, Java, Ruby, PHP, etc.) ----

    _GENERIC_PATTERNS: dict[str, list[tuple[re.Pattern[str], SymbolKind]]] = {}

    def _get_generic_patterns(self, language: str) -> list[tuple[re.Pattern[str], SymbolKind]]:
        if language in self._GENERIC_PATTERNS:
            return self._GENERIC_PATTERNS[language]
        patterns: list[tuple[re.Pattern[str], SymbolKind]] = []

        if language == "go":
            patterns = [
                (re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)"), SymbolKind.FUNCTION),
                (re.compile(r"^func\s+\(\w+\s+\*?(\w+)\)\s+(\w+)\s*\(([^)]*)\)"), SymbolKind.METHOD),
                (re.compile(r"^type\s+(\w+)\s+struct\s*\{"), SymbolKind.CLASS),
                (re.compile(r"^type\s+(\w+)\s+interface\s*\{"), SymbolKind.INTERFACE),
                (re.compile(r"^type\s+(\w+)\s+"), SymbolKind.TYPE),
                (re.compile(r"^var\s+(\w+)\s+"), SymbolKind.VARIABLE),
                (re.compile(r"^const\s+(\w+)\s+"), SymbolKind.CONSTANT),
                (re.compile(r'^(\w+)\s*:=\s*'), SymbolKind.VARIABLE),
                (re.compile(r'^import\s+"([^"]+)"'), SymbolKind.IMPORT),
            ]
        elif language == "rust":
            patterns = [
                (re.compile(r"^pub\s+(?:async\s+)?fn\s+(\w+)\s*\(([^)]*)\)"), SymbolKind.FUNCTION),
                (re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(([^)]*)\)"), SymbolKind.FUNCTION),
                (re.compile(r"^pub\s+struct\s+(\w+)"), SymbolKind.CLASS),
                (re.compile(r"^(?:pub\s+)?struct\s+(\w+)"), SymbolKind.CLASS),
                (re.compile(r"^pub\s+enum\s+(\w+)"), SymbolKind.ENUM),
                (re.compile(r"^(?:pub\s+)?enum\s+(\w+)"), SymbolKind.ENUM),
                (re.compile(r"^pub\s+trait\s+(\w+)"), SymbolKind.INTERFACE),
                (re.compile(r"^(?:pub\s+)?trait\s+(\w+)"), SymbolKind.INTERFACE),
                (re.compile(r"^(?:pub\s+)?type\s+(\w+)\s*="), SymbolKind.TYPE),
                (re.compile(r"^(?:pub\s+)?const\s+(\w+)\s*:"), SymbolKind.CONSTANT),
                (re.compile(r"^(?:pub\s+)?static\s+(\w+)\s*:"), SymbolKind.CONSTANT),
                (re.compile(r"^impl\s+(\w+)"), SymbolKind.CLASS),
                (re.compile(r"^use\s+([^;]+);"), SymbolKind.IMPORT),
                (re.compile(r"^(?:pub\s+)?mod\s+(\w+)"), SymbolKind.MODULE),
            ]
        elif language in ("java", "kotlin", "scala", "csharp"):
            patterns = [
                (re.compile(
                    r"^(?:public|private|protected|static|abstract|final|open|override|internal|"
                    r"sealed|data|suspend|companion|inline|crossinline|reified|lateinit|val|var)\s+)*"
                    r"(?:class|object|interface|enum|record|struct|data class)\s+(\w+)",
                ), SymbolKind.CLASS),
                (re.compile(
                    r"^(?:public|private|protected|static|abstract|final|suspended|open|override|"
                    r"internal|inline|suspend|lateinit|companion)\s+)*"
                    r"(?:suspend\s+)?(\w+)\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*(?:throws\s+[\w,\s]+)?\s*\{",
                ), SymbolKind.FUNCTION),
                (re.compile(r"^(?:public|private|protected|static|final|const|val)\s+(\w+)\s*[=:]"),
                 SymbolKind.CONSTANT),
                (re.compile(r"^import\s+([^;]+);"), SymbolKind.IMPORT),
                (re.compile(r"^package\s+([^;]+);"), SymbolKind.MODULE),
            ]
        elif language == "ruby":
            patterns = [
                (re.compile(r"^def\s+(?:self\.)?(\w+[?!]?)\s*\(([^)]*)\)"), SymbolKind.FUNCTION),
                (re.compile(r"^def\s+(?:self\.)?(\w+[?!]?)"), SymbolKind.FUNCTION),
                (re.compile(r"^class\s+(\w+)"), SymbolKind.CLASS),
                (re.compile(r"^module\s+(\w+)"), SymbolKind.MODULE),
                (re.compile(r"^[A-Z]\w+\s*=\s*"), SymbolKind.CONSTANT),
                (re.compile(r"^(\w+)\s*=\s*"), SymbolKind.VARIABLE),
                (re.compile(r"^require\s+['\"]([^'\"]+)['\"]"), SymbolKind.IMPORT),
                (re.compile(r"^(?:attr_reader|attr_writer|attr_accessor)\s+:(\w+)"), SymbolKind.PROPERTY),
            ]
        elif language == "php":
            patterns = [
                (re.compile(r"^(?:public|private|protected|static|abstract|final)\s+function\s+(\w+)\s*\(([^)]*)\)"),
                 SymbolKind.FUNCTION),
                (re.compile(r"^function\s+(\w+)\s*\(([^)]*)\)"), SymbolKind.FUNCTION),
                (re.compile(r"^(?:abstract\s+)?class\s+(\w+)"), SymbolKind.CLASS),
                (re.compile(r"^interface\s+(\w+)"), SymbolKind.INTERFACE),
                (re.compile(r"^trait\s+(\w+)"), SymbolKind.INTERFACE),
                (re.compile(r"^(?:const|public\s+const)\s+(\w+)\s*="), SymbolKind.CONSTANT),
                (re.compile(r"^\$(\w+)\s*="), SymbolKind.VARIABLE),
                (re.compile(r"^(?:use|require|include)(_once)?\s+['\"]([^'\"]+)['\"]"), SymbolKind.IMPORT),
                (re.compile(r"^(?:use)\s+([^;]+);"), SymbolKind.IMPORT),
                (re.compile(r"^namespace\s+([^;]+);"), SymbolKind.MODULE),
            ]
        elif language in ("c", "cpp"):
            patterns = [
                (re.compile(
                    r"^(?:(?:static|extern|inline|virtual|override|constexpr|template\s*<[^>]+>)\s+)*"
                    r"(?:void|int|char|float|double|long|short|unsigned|bool|auto|size_t|string|"
                    r"[A-Z]\w+)\s*\*?\s+(\w+)\s*\(([^)]*)\)",
                ), SymbolKind.FUNCTION),
                (re.compile(r"^(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+(\w+))?"),
                 SymbolKind.CLASS),
                (re.compile(r"^enum\s+(?:class\s+)?(\w+)"), SymbolKind.ENUM),
                (re.compile(r"^(?:typedef|using)\s+(\w+)"), SymbolKind.TYPE),
                (re.compile(r"^#define\s+(\w+)"), SymbolKind.CONSTANT),
                (re.compile(r"^#include\s+[<\"]([^>\"]+)[>\"]"), SymbolKind.IMPORT),
                (re.compile(r"^(?:const|constexpr)\s+\w+\s+(\w+)\s*="), SymbolKind.CONSTANT),
                (re.compile(r"^(?:namespace|using\s+namespace)\s+(\w+)"), SymbolKind.MODULE),
            ]
        elif language == "swift":
            patterns = [
                (re.compile(r"^(?:public|private|protected|internal|static|class|override|mutating|@\w+\s+)*func\s+(\w+)\s*\(([^)]*)\)"),
                 SymbolKind.FUNCTION),
                (re.compile(r"^(?:public|private|protected|internal|final|open|class)\s+class\s+(\w+)"),
                 SymbolKind.CLASS),
                (re.compile(r"^(?:public|private|protected|internal)\s+struct\s+(\w+)"),
                 SymbolKind.CLASS),
                (re.compile(r"^(?:public|private|protected|internal)\s+protocol\s+(\w+)"),
                 SymbolKind.INTERFACE),
                (re.compile(r"^(?:public|private|protected|internal)\s+enum\s+(\w+)"),
                 SymbolKind.ENUM),
                (re.compile(r"^let\s+(\w+)\s*[=:]"), SymbolKind.CONSTANT),
                (re.compile(r"^var\s+(\w+)\s*[=:]"), SymbolKind.VARIABLE),
                (re.compile(r"^import\s+(\w+)"), SymbolKind.IMPORT),
            ]
        else:
            # Ultimate fallback
            patterns = [
                (re.compile(r"^(?:function|def|func|fn|sub)\s+(\w+)\s*\("), SymbolKind.FUNCTION),
                (re.compile(r"^(?:class|struct|interface|protocol)\s+(\w+)"), SymbolKind.CLASS),
                (re.compile(r"^(?:const|constant|final|let)\s+(\w+)\s*[=:]"), SymbolKind.CONSTANT),
                (re.compile(r"^(?:var|let|mut|dim)\s+(\w+)\s*[=:]"), SymbolKind.VARIABLE),
                (re.compile(r"^(?:import|use|require|include|from)\s+"), SymbolKind.IMPORT),
            ]

        self._GENERIC_PATTERNS[language] = patterns
        return patterns

    def _extract_symbols_generic(
        self, filepath: str, content: str, language: str
    ) -> list[Symbol]:
        symbols: list[Symbol] = []
        module_name = self._module_name(filepath)
        patterns = self._get_generic_patterns(language)

        for pattern, kind in patterns:
            for m in pattern.finditer(content):
                ln = content[:m.start()].count("\n") + 1
                name = m.group(1) if m.lastindex else m.group(0)
                sig = m.group(0).strip()
                end_ln = content[:m.end()].count("\n") + 1
                sym = Symbol(
                    name=name, kind=kind, file_path=filepath,
                    line=ln, end_line=end_ln,
                    signature=sig, parent="",
                    module=module_name, language=language,
                    hash=self._content_string_hash(sig),
                )
                symbols.append(sym)

        return symbols

    # ------------------------------------------------------------------
    # Reference graph
    # ------------------------------------------------------------------

    async def _build_references(self) -> None:
        """Build a reference graph by scanning file contents for symbol usages."""
        self._references.clear()
        self._reverse_refs.clear()
        self._callers.clear()
        self._callees.clear()

        all_symbol_names: set[str] = set(self._symbols_by_name.keys())
        if not all_symbol_names:
            return

        # Build a regex that matches any known symbol name (word boundary)
        # We process in batches to avoid catastrophic regex backtracking
        name_list = sorted(all_symbol_names, key=len, reverse=True)
        batch_size = 200
        for i in range(0, len(name_list), batch_size):
            batch = name_list[i:i + batch_size]
            escaped = [re.escape(n) for n in batch if n.isidentifier() or (n and n[0].isalpha())]
            if not escaped:
                continue
            try:
                combined = re.compile(r"\b(?:" + "|".join(escaped) + r")\b")
            except re.error:
                continue
            for filepath, syms in self._symbols_by_file.items():
                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                        content = fh.read()
                except OSError:
                    continue
                lines = content.splitlines()
                for match in combined.finditer(content):
                    ln = content[:match.start()].count("\n") + 1
                    matched_name = match.group(0)
                    # Skip the definition line itself
                    is_def = any(
                        s.name == matched_name and s.line == ln and s.file_path == filepath
                        for s in syms
                    )
                    if is_def:
                        continue
                    ref_kind = self._classify_reference(matched_name, lines[ln - 1] if ln <= len(lines) else "")
                    ref = Reference(
                        symbol_name=matched_name, file_path=filepath,
                        line=ln, kind=ref_kind,
                        context=(lines[ln - 1].strip() if ln <= len(lines) else ""),
                    )
                    self._references[matched_name].append(ref)
                    self._reverse_refs[f"{filepath}:{ln}"].append(ref)

        # Build caller / callee maps
        for sym_key, refs in self._reverse_refs.items():
            caller_syms = self._resolve_symbols_at(
                refs[0].file_path if refs else "", int(sym_key.split(":")[-1]) if refs and ":" in sym_key else 0
            )
            for ref in refs:
                if ref.kind == ReferenceKind.CALL:
                    for cs in caller_syms:
                        for clee in self._symbols_by_name.get(ref.symbol_name, []):
                            if clee.file_path != cs.file_path or clee.line != cs.line:
                                if cs not in self._callers[ref.symbol_name]:
                                    self._callers[ref.symbol_name].append(cs)
                                ckey = self._sym_key(cs)
                                if clee not in self._callees.get(ckey, []):
                                    self._callees.setdefault(ckey, []).append(clee)

    def _classify_reference(self, name: str, line_text: str) -> ReferenceKind:
        """Heuristic classification of a reference kind based on surrounding text."""
        stripped = line_text.strip()
        # Import detection
        if re.match(r"^(?:import|from|use|require|include)\b", stripped):
            return ReferenceKind.IMPORT
        # Call detection
        if re.search(rf"\b{name}\s*\(", stripped):
            return ReferenceKind.CALL
        # Inheritance
        if re.match(r"^(?:class|struct)\s+\w+\s*[:(].*" + re.escape(name), stripped):
            return ReferenceKind.INHERITANCE
        # Type annotation
        if re.search(rf"(?::\s*{re.escape(name)}|<\s*{re.escape(name)}|extends\s+{re.escape(name)}|implements\s+{re.escape(name)})", stripped):
            return ReferenceKind.TYPE_ANNOTATION
        # Assignment
        if re.search(rf"=\s*{re.escape(name)}\b", stripped):
            return ReferenceKind.ASSIGNMENT
        return ReferenceKind.REFERENCE

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    async def get_symbol(self, name: str, file_path: str | None = None) -> Symbol | None:
        candidates = self._symbols_by_name.get(name, [])
        if file_path:
            for s in candidates:
                if s.file_path == file_path:
                    return s
        return candidates[0] if candidates else None

    async def get_symbols_in_file(self, filepath: str) -> list[Symbol]:
        return list(self._symbols_by_file.get(filepath, []))

    async def get_symbols_by_kind(self, kind: SymbolKind) -> list[Symbol]:
        return list(self._symbols_by_kind.get(kind, []))

    async def find_references(self, symbol_name: str) -> list[Reference]:
        return list(self._references.get(symbol_name, []))

    async def find_callers(self, symbol_name: str) -> list[Symbol]:
        return list(self._callers.get(symbol_name, []))

    async def find_callees(self, symbol_name: str) -> list[Symbol]:
        sym = await self.get_symbol(symbol_name)
        if not sym:
            return []
        return list(self._callees.get(self._sym_key(sym), []))

    async def get_definition(self, symbol_name: str) -> Symbol | None:
        """Go-to-definition: find where a symbol is defined."""
        candidates = self._symbols_by_name.get(symbol_name, [])
        for sym in candidates:
            if sym.kind in (SymbolKind.FUNCTION, SymbolKind.CLASS, SymbolKind.METHOD,
                            SymbolKind.CONSTANT, SymbolKind.ENUM, SymbolKind.INTERFACE,
                            SymbolKind.TYPE):
                return sym
        return candidates[0] if candidates else None

    async def get_import_chain(self, filepath: str, depth: int = 3) -> list[dict]:
        """Follow import chains from a file, returning a list of chain entries."""
        visited: set[str] = {filepath}
        chains: list[dict] = []
        queue: list[tuple[str, int, list[str]]] = [(filepath, 0, [filepath])]

        while queue:
            current, level, path = queue.pop(0)
            if level > depth:
                continue
            imports = [
                s for s in self._symbols_by_file.get(current, [])
                if s.kind == SymbolKind.IMPORT
            ]
            for imp in imports:
                import_module = imp.metadata.get("import_module") or imp.metadata.get("import_path", "")
                if not import_module:
                    continue
                entry = {
                    "from_file": current,
                    "import_name": imp.name,
                    "import_module": import_module,
                    "depth": level,
                    "path": list(path),
                }
                chains.append(entry)

                # Try to resolve to actual file
                resolved = self._resolve_import_to_file(current, import_module, imp.name)
                if resolved and resolved not in visited:
                    visited.add(resolved)
                    queue.append((resolved, level + 1, path + [resolved]))

        return chains

    def _resolve_import_to_file(self, from_file: str, module: str, name: str) -> str | None:
        """Heuristic resolution of an import module/path to a project file."""
        if not module:
            return None
        # Python-style: replace dots with path separators
        parts = module.split(".")
        candidate_dirs = [
            os.path.dirname(from_file),
            self.project_path,
        ]
        for base in candidate_dirs:
            # Try direct file
            for ext in (".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".php"):
                fpath = os.path.join(base, *parts) + ext
                if os.path.isfile(fpath):
                    return fpath
            # Try __init__.py
            init_path = os.path.join(base, *parts, "__init__.py")
            if os.path.isfile(init_path):
                return init_path
            # Try index.js / index.ts
            for idx in ("index.js", "index.ts", "index.jsx", "index.tsx"):
                idx_path = os.path.join(base, *parts, idx)
                if os.path.isfile(idx_path):
                    return idx_path
        return None

    # ------------------------------------------------------------------
    # TF-IDF Semantic Search
    # ------------------------------------------------------------------

    async def _rebuild_tfidf(self) -> None:
        """Recompute TF-IDF vectors for all symbols."""
        if not self._symbols:
            self._vocabulary.clear()
            self._idf.clear()
            self._tfidf_vectors.clear()
            return

        # Build documents: one per symbol, from name + signature + docstring + parent context
        documents: dict[str, list[str]] = {}
        for key, sym in self._symbols.items():
            parts = [sym.name, sym.signature, sym.docstring, sym.parent, sym.module, sym.language]
            # Add parent's name and docstring for context
            if sym.parent:
                parent_syms = [s for s in self._symbols_by_name.get(sym.parent, [])
                               if s.file_path == sym.file_path]
                for ps in parent_syms:
                    parts.append(ps.name)
                    parts.append(ps.docstring)
            text = " ".join(p for p in parts if p)
            documents[key] = _tokenize(text)

        # Build vocabulary
        all_terms: set[str] = set()
        for tokens in documents.values():
            all_terms.update(tokens)
        self._vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
        vocab_size = len(self._vocabulary)

        # Compute IDF
        n = len(documents)
        df: Counter[str] = Counter()
        for tokens in documents.values():
            unique_terms = set(tokens)
            for t in unique_terms:
                df[t] += 1
        self._idf = {}
        for term in self._vocabulary:
            self._idf[term] = math.log((n + 1) / (1 + df.get(term, 0))) + 1

        # Compute TF-IDF vectors
        self._tfidf_vectors.clear()
        for key, tokens in documents.items():
            tf: Counter[str] = Counter(tokens)
            vec = [0.0] * vocab_size
            for term, count in tf.items():
                idx = self._vocabulary.get(term)
                if idx is not None:
                    vec[idx] = count * self._idf.get(term, 0.0)
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            self._tfidf_vectors[key] = vec

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute a TF-IDF vector for arbitrary query text."""
        if not self._vocabulary:
            return []
        tokens = _tokenize(text)
        tf: Counter[str] = Counter(tokens)
        vocab_size = len(self._vocabulary)
        vec = [0.0] * vocab_size
        for term, count in tf.items():
            idx = self._vocabulary.get(term)
            if idx is not None:
                idf_val = self._idf.get(term, 1.0)
                vec[idx] = count * idf_val
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def search(
        self,
        query: str,
        limit: int = 20,
        kind: SymbolKind | None = None,
        file_filter: str | None = None,
    ) -> list[SearchResult]:
        """Semantic + keyword search over all indexed symbols."""
        query_lower = query.lower()
        query_vec = self._compute_embedding(query)
        scored: list[tuple[float, Symbol, str, list[str]]] = []

        for key, sym in self._symbols.items():
            if kind and sym.kind != kind:
                continue
            if file_filter and file_filter not in sym.file_path:
                continue

            # Semantic score (TF-IDF cosine)
            sym_vec = self._tfidf_vectors.get(key)
            semantic_score = self._cosine_similarity(query_vec, sym_vec) if sym_vec else 0.0

            # Keyword score (exact substring matching)
            keyword_score = 0.0
            highlights: list[str] = []
            searchable = f"{sym.name} {sym.signature} {sym.docstring} {sym.parent} {sym.module}".lower()
            if query_lower in sym.name.lower():
                keyword_score += 3.0
                highlights.append(sym.name)
            if query_lower in searchable:
                keyword_score += 1.0
            # Check individual query terms
            for term in query_lower.split():
                if term in sym.name.lower():
                    keyword_score += 0.5
                if term in sym.signature.lower():
                    keyword_score += 0.3
                if term in sym.docstring.lower():
                    keyword_score += 0.2

            # Combined score: weighted blend
            combined = 0.6 * semantic_score + 0.4 * min(keyword_score, 5.0)
            match_type = "semantic" if semantic_score > keyword_score else "keyword"
            if combined > 0.05:
                scored.append((combined, sym, match_type, highlights))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SearchResult] = []
        for score, sym, mtype, highlights in scored[:limit]:
            # Build context snippet
            ctx = self._get_symbol_context(sym)
            results.append(SearchResult(
                symbol=sym, score=round(score, 4),
                match_type=mtype, context=ctx, highlights=highlights,
            ))
        return results

    def _get_symbol_context(self, sym: Symbol) -> str:
        """Read a few lines around a symbol for context."""
        try:
            with open(sym.file_path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            start = max(0, sym.line - 2)
            end = min(len(lines), sym.end_line + 3)
            return "".join(lines[start:end]).strip()
        except OSError:
            return sym.signature

    # ------------------------------------------------------------------
    # Smart context selection for LLM
    # ------------------------------------------------------------------

    async def get_context_for_query(
        self, query: str, max_tokens: int = 50000
    ) -> list[dict]:
        """Select the most relevant symbols/files to include in LLM context."""
        search_results = await self.search(query, limit=50)
        budget = max_tokens
        selected: list[dict] = []

        # Track included files to avoid duplicates
        included_files: set[str] = set()

        # Phase 1: Add top search results
        for sr in search_results:
            sym = sr.symbol
            ctx = self._get_symbol_context(sym)
            tokens = self._token_estimate(ctx)
            if budget - tokens < 0:
                break
            selected.append({
                "type": "symbol",
                "symbol": sym.name,
                "kind": sym.kind.value,
                "file": sym.file_path,
                "line": sym.line,
                "score": sr.score,
                "context": ctx,
                "tokens": tokens,
                "reason": f"semantic_match ({sr.match_type})",
            })
            budget -= tokens
            included_files.add(sym.file_path)

        # Phase 2: Add related symbols (import chain, callers/callees)
        related_names: set[str] = set()
        for sr in search_results[:10]:
            sym = sr.symbol
            # Get imports from same file
            for s in self._symbols_by_file.get(sym.file_path, []):
                if s.kind == SymbolKind.IMPORT:
                    related_names.add(s.name)
            # Get callers/callees
            for caller in self._callers.get(sym.name, []):
                related_names.add(caller.name)
            for callee_sym_key, callees in self._callees.items():
                for c in callees:
                    if c.name == sym.name:
                        ck_sym = self._symbols.get(callee_sym_key)
                        if ck_sym:
                            related_names.add(ck_sym.name)

        for name in related_names:
            sym = await self.get_symbol(name)
            if not sym or sym.file_path in included_files:
                continue
            ctx = self._get_symbol_context(sym)
            tokens = self._token_estimate(ctx)
            if budget - tokens < 0:
                break
            selected.append({
                "type": "related_symbol",
                "symbol": sym.name,
                "kind": sym.kind.value,
                "file": sym.file_path,
                "line": sym.line,
                "context": ctx,
                "tokens": tokens,
                "reason": "graph_related",
            })
            budget -= tokens
            included_files.add(sym.file_path)

        # Phase 3: Add important files not yet included (entry points, config)
        priority_files = self._get_priority_files()
        for pf in priority_files:
            if pf in included_files:
                continue
            try:
                with open(pf, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            tokens = self._token_estimate(content)
            if budget - tokens < 0:
                # Truncate
                lines = content.splitlines()
                truncated = "\n".join(lines[:max(1, int(len(lines) * budget / max(tokens, 1)))])
                tokens = self._token_estimate(truncated)
                if tokens > budget:
                    continue
                selected.append({
                    "type": "file_truncated",
                    "file": pf,
                    "context": truncated,
                    "tokens": tokens,
                    "reason": "priority_file",
                })
                budget -= tokens
                included_files.add(pf)
            else:
                selected.append({
                    "type": "file",
                    "file": pf,
                    "context": content,
                    "tokens": tokens,
                    "reason": "priority_file",
                })
                budget -= tokens
                included_files.add(pf)

        return selected

    async def get_context_for_position(
        self, filepath: str, line: int, max_tokens: int = 30000
    ) -> list[dict]:
        """Select relevant context for a cursor position in a file."""
        selected: list[dict] = []
        budget = max_tokens

        # Phase 1: enclosing file symbols
        file_syms = self._symbols_by_file.get(filepath, [])
        # Sort by proximity to the cursor line
        file_syms_sorted = sorted(file_syms, key=lambda s: min(abs(s.line - line), abs(s.end_line - line)))

        for sym in file_syms_sorted:
            ctx = self._get_symbol_context(sym)
            tokens = self._token_estimate(ctx)
            if budget - tokens < 0:
                break
            distance = abs(sym.line - line)
            selected.append({
                "type": "file_symbol",
                "symbol": sym.name,
                "kind": sym.kind.value,
                "line": sym.line,
                "end_line": sym.end_line,
                "context": ctx,
                "tokens": tokens,
                "reason": f"proximity (distance={distance})",
            })
            budget -= tokens

        # Phase 2: symbols referenced at this line
        refs_at_line = self._reverse_refs.get(f"{filepath}:{line}", [])
        for ref in refs_at_line:
            defn = await self.get_definition(ref.symbol_name)
            if not defn or defn.file_path == filepath:
                continue
            ctx = self._get_symbol_context(defn)
            tokens = self._token_estimate(ctx)
            if budget - tokens < 0:
                break
            selected.append({
                "type": "definition",
                "symbol": defn.name,
                "kind": defn.kind.value,
                "file": defn.file_path,
                "line": defn.line,
                "context": ctx,
                "tokens": tokens,
                "reason": f"referenced_at_line:{line}",
            })
            budget -= tokens

        # Phase 3: parent scope context (if inside a class/function, show class def)
        enclosing = self._resolve_symbols_at(filepath, line)
        for enc in enclosing:
            if enc.kind in (SymbolKind.CLASS, SymbolKind.FUNCTION, SymbolKind.METHOD):
                ctx = self._get_symbol_context(enc)
                tokens = self._token_estimate(ctx)
                if budget - tokens < 0:
                    break
                # Avoid duplicate if already added
                already = any(
                    s.get("symbol") == enc.name and s.get("line") == enc.line
                    for s in selected
                )
                if not already:
                    selected.append({
                        "type": "enclosing_scope",
                        "symbol": enc.name,
                        "kind": enc.kind.value,
                        "line": enc.line,
                        "context": ctx,
                        "tokens": tokens,
                        "reason": "enclosing_scope",
                    })
                    budget -= tokens

        return selected

    def _get_priority_files(self) -> list[str]:
        """Return entry-point / config files sorted by importance."""
        priority_names = [
            "README.md", "README", "readme.md",
            "package.json", "pyproject.toml", "setup.py", "setup.cfg",
            "Cargo.toml", "go.mod", "go.sum",
            "Makefile", "CMakeLists.txt",
            "tsconfig.json", ".eslintrc", ".prettierrc",
            "docker-compose.yml", "Dockerfile",
            ".env", ".env.example",
            "main.py", "main.go", "main.rs", "main.java", "main.ts", "main.js",
            "index.py", "index.ts", "index.js", "index.tsx", "index.jsx",
            "app.py", "app.ts", "app.js", "app.go", "app.rs",
            "manage.py", "cli.py",
        ]
        found: list[str] = []
        for name in priority_names:
            fpath = os.path.join(self.project_path, name)
            if os.path.isfile(fpath):
                found.append(fpath)
        return found

    # ------------------------------------------------------------------
    # Cross-file type inference
    # ------------------------------------------------------------------

    async def infer_type(self, symbol_name: str, file_path: str) -> str | None:
        """Infer the type of a symbol by tracing assignments, function returns, etc."""
        sym = await self.get_symbol(symbol_name, file_path)
        if not sym:
            return None

        # Check explicit type annotation
        type_ann = sym.metadata.get("type_annotation")
        if type_ann:
            return type_ann
        ret_type = sym.metadata.get("return_type")
        if ret_type:
            return ret_type

        # Trace variable assignments
        if sym.kind in (SymbolKind.VARIABLE, SymbolKind.CONSTANT, SymbolKind.PROPERTY):
            # Look for the assignment line
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
                if sym.line <= len(lines):
                    line = lines[sym.line - 1].strip()
                    # Pattern: var = SomeClass(...)
                    m = re.match(rf"^{re.escape(sym.name)}\s*=\s*(\w+)(?:\(|\.|$)", line)
                    if m:
                        return m.group(1)
                    # Pattern: var = "string"
                    if "= \"" in line or "= '" in line:
                        return "str"
                    # Pattern: var = 123
                    m = re.match(rf"^{re.escape(sym.name)}\s*=\s*(\d+\.?\d*)", line)
                    if m:
                        return "int" if "." not in m.group(1) else "float"
                    # Pattern: var = True/False/None
                    if re.match(rf"^{re.escape(sym.name)}\s*=\s*(True|False|None)$", line):
                        val = re.match(rf"^{re.escape(sym.name)}\s*=\s*(True|False|None)$", line)
                        return val.group(1).lower() if val else None
                    # Pattern: var = []
                    if "= []" in line or "= list(" in line:
                        return "list"
                    # Pattern: var = {}
                    if "= {}" in line or "= dict(" in line:
                        return "dict"
                    # Pattern: var = function_call()
                    m = re.match(rf"^{re.escape(sym.name)}\s*=\s*(\w+)\(", line)
                    if m:
                        func_name = m.group(1)
                        # Look up the function's return type
                        func_sym = await self.get_definition(func_name)
                        if func_sym:
                            func_ret = func_sym.metadata.get("return_type")
                            if func_ret:
                                return func_ret
                            return f"{func_name}()"  # fallback
            except OSError:
                pass

        # Trace function return types
        if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                lines = content.splitlines()
                # Collect return statements
                for i in range(sym.line, min(sym.end_line + 1, len(lines) + 1)):
                    line = lines[i - 1].strip()
                    ret_m = re.match(r"^return\s+(.+)$", line)
                    if ret_m:
                        ret_expr = ret_m.group(1).strip()
                        # return "string"
                        if ret_expr.startswith('"') or ret_expr.startswith("'"):
                            return "str"
                        # return 123
                        if re.match(r"^\d+\.?\d*$", ret_expr):
                            return "int" if "." not in ret_expr else "float"
                        # return True/False/None
                        if ret_expr in ("True", "False", "None"):
                            return ret_expr.lower()
                        # return []
                        if ret_expr == "[]" or ret_expr.startswith("list("):
                            return "list"
                        # return {}
                        if ret_expr == "{}" or ret_expr.startswith("dict("):
                            return "dict"
                        # return SomeClass(...)
                        ret_cls_m = re.match(r"^(\w+)\(", ret_expr)
                        if ret_cls_m:
                            return ret_cls_m.group(1)
                        # return variable
                        ret_var_m = re.match(r"^(\w+)$", ret_expr)
                        if ret_var_m:
                            var_type = await self.infer_type(ret_var_m.group(1), file_path)
                            if var_type:
                                return var_type
                        # return self.var
                        self_m = re.match(r"^self\.(\w+)$", ret_expr)
                        if self_m:
                            attr = self_m.group(1)
                            if sym.parent:
                                parent_syms = self._symbols_by_name.get(sym.parent, [])
                                for ps in parent_syms:
                                    if ps.file_path == file_path:
                                        attr_syms = [
                                            s for s in self._symbols_by_file.get(file_path, [])
                                            if s.name == attr and s.parent == sym.parent
                                        ]
                                        for attr_sym in attr_syms:
                                            t = await self.infer_type(attr_sym.name, file_path)
                                            if t:
                                                return t
                        # return self
                        if ret_expr == "self":
                            return sym.parent or "Self"
                        break  # first return is often representative
            except OSError:
                pass

        return None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Return comprehensive codebase statistics."""
        total_files = len(self._file_indices)
        total_symbols = len(self._symbols)
        lang_counts: Counter[str] = Counter()
        kind_counts: Counter[str] = Counter()
        doc_count = 0
        total_lines = 0
        complexity_scores: list[int] = []

        for fi in self._file_indices.values():
            lang_counts[fi.language] += 1
            total_lines += fi.line_count

        for sym in self._symbols.values():
            kind_counts[sym.kind.value] += 1
            if sym.docstring:
                doc_count += 1
            # Simple complexity heuristic: lines of code for functions/methods
            if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                loc = sym.end_line - sym.line + 1
                complexity_scores.append(loc)

        avg_complexity = (
            sum(complexity_scores) / len(complexity_scores)
            if complexity_scores else 0
        )
        doc_coverage = (doc_count / total_symbols * 100) if total_symbols else 0

        # Compute total reference count
        total_refs = sum(len(refs) for refs in self._references.values())

        return {
            "total_files": total_files,
            "total_symbols": total_symbols,
            "total_lines": total_lines,
            "total_references": total_refs,
            "languages": dict(lang_counts),
            "symbol_kinds": dict(kind_counts),
            "average_complexity": round(avg_complexity, 1),
            "documentation_coverage": round(doc_coverage, 1),
            "files_with_symbols": len(self._symbols_by_file),
            "unique_symbol_names": len(self._symbols_by_name),
            "vocabulary_size": len(self._vocabulary),
            "index_size_bytes": os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0,
        }

    # ------------------------------------------------------------------
    # Call graph
    # ------------------------------------------------------------------

    async def get_call_graph(self, symbol_name: str, depth: int = 3) -> dict:
        """Build a recursive call graph starting from a symbol."""
        sym = await self.get_symbol(symbol_name)
        if not sym:
            return {}

        graph: dict[str, Any] = {
            "name": sym.name,
            "kind": sym.kind.value,
            "file": sym.file_path,
            "line": sym.line,
            "children": [],
        }
        visited: set[str] = {self._sym_key(sym)}
        await self._build_call_graph_recursive(sym, graph["children"], depth, visited)
        return graph

    async def _build_call_graph_recursive(
        self, sym: Symbol, children: list[dict], depth: int, visited: set[str],
    ) -> None:
        if depth <= 0:
            return
        callees = self._callees.get(self._sym_key(sym), [])
        for callee in callees:
            ckey = self._sym_key(callee)
            if ckey in visited:
                continue
            visited.add(ckey)
            node: dict[str, Any] = {
                "name": callee.name,
                "kind": callee.kind.value,
                "file": callee.file_path,
                "line": callee.line,
                "children": [],
            }
            children.append(node)
            await self._build_call_graph_recursive(callee, node["children"], depth - 1, visited)

    async def get_dependency_graph(self) -> dict[str, list[str]]:
        """Build a module-level dependency graph from imports."""
        graph: dict[str, list[str]] = {}
        for filepath, syms in self._symbols_by_file.items():
            deps: set[str] = set()
            for sym in syms:
                if sym.kind == SymbolKind.IMPORT:
                    import_module = sym.metadata.get("import_module") or sym.metadata.get("import_path", "")
                    if import_module:
                        resolved = self._resolve_import_to_file(filepath, import_module, sym.name)
                        if resolved:
                            deps.add(resolved)
                        else:
                            deps.add(import_module)
            graph[filepath] = sorted(deps)
        return graph

    async def find_dead_symbols(self) -> list[Symbol]:
        """Find symbols that are defined but never referenced elsewhere."""
        dead: list[Symbol] = []
        for sym in self._symbols.values():
            if sym.kind in (SymbolKind.MODULE, SymbolKind.IMPORT):
                continue
            refs = self._references.get(sym.name, [])
            # Filter out self-references (the definition itself)
            external_refs = [
                r for r in refs
                if not (r.file_path == sym.file_path and r.line == sym.line)
            ]
            if not external_refs:
                dead.append(sym)
        return dead

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _language_from_extension(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        return LANGUAGE_EXTENSIONS.get(ext, "")

    @staticmethod
    def _file_content_hash(filepath: str) -> str:
        h = hashlib.sha256()
        try:
            with open(filepath, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()

    @staticmethod
    def _content_string_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _module_name(self, filepath: str) -> str:
        rel = os.path.relpath(filepath, self.project_path)
        base, _ = os.path.splitext(rel)
        return base.replace(os.sep, ".")

    def _is_supported_file(self, filepath: str) -> bool:
        ext = os.path.splitext(filepath)[1].lower()
        return ext in LANGUAGE_EXTENSIONS

    @staticmethod
    def _token_estimate(text: str) -> int:
        """Rough token count: ~4 characters per token for English/code."""
        return max(1, len(text) // 4)

    async def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# Utility: simple glob-to-regex for directory exclusion
# ---------------------------------------------------------------------------

def fnmatch_simple(name: str, pattern: str) -> bool:
    """Very simple glob matching (only * supported, no ? or [])."""
    if "*" not in pattern:
        return name == pattern
    parts = pattern.split("*")
    if len(parts) == 2:
        return name.startswith(parts[0]) and name.endswith(parts[1])
    return pattern.replace("*", "") in name


# ---------------------------------------------------------------------------
# Convenience: async main for CLI usage
# ---------------------------------------------------------------------------

async def _main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Codebase Indexer")
    parser.add_argument("path", help="Project root path")
    parser.add_argument("--db", default=None, help="Custom DB path")
    parser.add_argument("--query", "-q", default=None, help="Search query")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--list-symbols", action="store_true", help="List all symbols")
    parser.add_argument("--find-dead", action="store_true", help="Find dead symbols")
    parser.add_argument("--max", type=int, default=20, help="Max results")
    args = parser.parse_args()

    indexer = CodebaseIndexer(args.path, db_path=args.db)
    await indexer.initialize()
    stats = await indexer.index_project()
    print(f"Indexed {stats.files_indexed} files, {stats.symbols_extracted} symbols "
          f"in {stats.duration_seconds:.2f}s")

    if args.query:
        results = await indexer.search(args.query, limit=args.max)
        print(f"\nSearch results for '{args.query}':")
        for r in results:
            print(f"  [{r.score:.3f}] {r.match_type} | {r.symbol.kind.value} "
                  f"{r.symbol.name} ({r.symbol.file_path}:{r.symbol.line})")
            if r.highlights:
                print(f"         highlights: {', '.join(r.highlights)}")

    if args.stats:
        st = await indexer.get_stats()
        print(f"\nCodebase Statistics:")
        print(json.dumps(st, indent=2, default=str))

    if args.list_symbols:
        for key, sym in sorted(indexer._symbols.items()):
            print(f"  {sym.kind.value:12s} {sym.name:40s} {sym.file_path}:{sym.line}")

    if args.find_dead:
        dead = await indexer.find_dead_symbols()
        print(f"\nDead symbols ({len(dead)}):")
        for sym in dead[:50]:
            print(f"  {sym.kind.value:12s} {sym.name:40s} {sym.file_path}:{sym.line}")

    await indexer.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
