"""
Concrete knowledge source implementations.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge.base import KnowledgeSource

logger = logging.getLogger(__name__)


class StringKnowledgeSource(KnowledgeSource):
    """
    Knowledge source from a plain string.
    
    Args:
        content: The text content to ingest.
        metadata: Optional metadata dictionary.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks in characters.
    """
    
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ):
        self.content = content
        self._metadata = metadata or {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load(self) -> List[str]:
        return self._split_text(self.content)
    
    @property
    def source_type(self) -> str:
        return "string"
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {"char_count": len(self.content), **self._metadata}
    
    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        return [c for c in chunks if c.strip()]


class TextFileKnowledgeSource(KnowledgeSource):
    """
    Knowledge source from a text file.
    
    Args:
        file_path: Path to the text file.
        encoding: File encoding.
        chunk_size: Maximum characters per chunk.
    """
    
    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        chunk_size: int = 2000,
    ):
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.chunk_size = chunk_size
    
    def load(self) -> List[str]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        content = self.file_path.read_text(encoding=self.encoding, errors="replace")
        return StringKnowledgeSource(
            content, metadata={"file": str(self.file_path)}, chunk_size=self.chunk_size
        ).load()
    
    @property
    def source_type(self) -> str:
        return "text_file"
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {"file": str(self.file_path)}


class JSONKnowledgeSource(KnowledgeSource):
    """
    Knowledge source from a JSON file.
    
    Flattens the JSON structure into text chunks.
    """
    
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = Path(file_path)
        self.encoding = encoding
    
    def load(self) -> List[str]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        content = self.file_path.read_text(encoding=self.encoding)
        data = json.loads(content)
        return [json.dumps(item, indent=2) if not isinstance(item, str) else item
                for item in self._flatten(data)]
    
    def _flatten(self, data: Any, prefix: str = "") -> List[Any]:
        items = []
        if isinstance(data, dict):
            for k, v in data.items():
                items.extend(self._flatten(v, f"{prefix}.{k}"))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                items.extend(self._flatten(v, f"{prefix}[{i}]"))
        else:
            items.append(data)
        return items
    
    @property
    def source_type(self) -> str:
        return "json"
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {"file": str(self.file_path)}


class CSVKnowledgeSource(KnowledgeSource):
    """Knowledge source from a CSV file."""
    
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = Path(file_path)
        self.encoding = encoding
    
    def load(self) -> List[str]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        content = self.file_path.read_text(encoding=self.encoding)
        reader = csv.DictReader(io.StringIO(content))
        return [
            json.dumps(row) for row in reader
        ]
    
    @property
    def source_type(self) -> str:
        return "csv"
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {"file": str(self.file_path)}


class ExcelKnowledgeSource(KnowledgeSource):
    """Knowledge source from an Excel file (.xlsx)."""
    
    def __init__(self, file_path: str, sheet_name: Optional[str] = None):
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name
    
    def load(self) -> List[str]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        try:
            import openpyxl
            wb = openpyxl.load_workbook(self.file_path, read_only=True)
            sheet = wb[self.sheet_name] if self.sheet_name else wb.active
            rows = []
            for row in sheet.iter_rows(values_only=True):
                rows.append(json.dumps([str(c) for c in row]))
            wb.close()
            return rows
        except ImportError:
            logger.warning("openpyxl not installed; trying csv fallback for xlsx")
            return []
    
    @property
    def source_type(self) -> str:
        return "excel"
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {"file": str(self.file_path), "sheet": self.sheet_name}


class PDFKnowledgeSource(KnowledgeSource):
    """Knowledge source from a PDF file."""
    
    def __init__(self, file_path: str, chunk_size: int = 2000):
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
    
    def load(self) -> List[str]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(self.file_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            full_text = "\n".join(text_parts)
            return StringKnowledgeSource(
                full_text, metadata={"file": str(self.file_path)},
                chunk_size=self.chunk_size,
            ).load()
        except ImportError:
            logger.warning("PyMuPDF not installed; PDF extraction unavailable")
            return []
    
    @property
    def source_type(self) -> str:
        return "pdf"
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return {"file": str(self.file_path)}
