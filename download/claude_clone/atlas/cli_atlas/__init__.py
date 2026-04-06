"""
Atlas CLI/UI Component — Interactive terminal interface for Claude Clone.

Provides a premium, feature-rich CLI experience with:
- Interactive TUI built on prompt_toolkit and rich
- Subcommand routing via argparse
- Configuration management with YAML support
- Multi-profile support
- Theme/skin engine
- Model, provider, tool, skill, gateway, cron, and MCP management
- Setup wizard and diagnostic tools
"""

__version__ = "1.0.0"
__author__ = "Atlas Agent"
__description__ = "Interactive CLI/UI for Claude Clone"

from atlas.cli_atlas.main import AtlasCLI
from atlas.cli_atlas.tui import AtlasTUI
from atlas.cli_atlas.config_manager import ConfigManager
from atlas.cli_atlas.banner import Banner
from atlas.cli_atlas.skin_engine import SkinEngine
from atlas.cli_atlas.profiles import ProfileManager

__all__ = [
    "AtlasCLI",
    "AtlasTUI",
    "ConfigManager",
    "Banner",
    "SkinEngine",
    "ProfileManager",
    "__version__",
]
