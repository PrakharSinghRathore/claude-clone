"""
Knowledge — external knowledge source management.

Provides a unified interface for ingesting and querying knowledge from
multiple source types (PDF, CSV, Excel, JSON, text files, strings).
"""

from knowledge.base import KnowledgeBase, KnowledgeSource
from knowledge.sources import (
    StringKnowledgeSource,
    TextFileKnowledgeSource,
    JSONKnowledgeSource,
    CSVKnowledgeSource,
    ExcelKnowledgeSource,
    PDFKnowledgeSource,
)

__all__ = [
    "KnowledgeBase",
    "KnowledgeSource",
    "StringKnowledgeSource",
    "TextFileKnowledgeSource",
    "JSONKnowledgeSource",
    "CSVKnowledgeSource",
    "ExcelKnowledgeSource",
    "PDFKnowledgeSource",
]
