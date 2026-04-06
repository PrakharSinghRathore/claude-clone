"""
Atlas Skills System — Skill management, loading, execution, and registry.

Provides a framework for defining, discovering, validating, and
executing agent skills with dependency resolution, versioning,
and a self-improving loop where the agent creates skills from
complex tasks.
"""

from .manager import SkillManager
from .loader import SkillLoader
from .registry import SkillRegistry
from .executor import SkillExecutor

__all__ = [
    "SkillManager",
    "SkillLoader",
    "SkillRegistry",
    "SkillExecutor",
]
