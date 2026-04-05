"""
Hermes CLI/UI Component — Interactive terminal interface for Claude Clone.

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
__author__ = "Hermes Agent"
__description__ = "Interactive CLI/UI for Claude Clone"

from hermes.cli_hermes.main import HermesCLI
from hermes.cli_hermes.tui import HermesTUI
from hermes.cli_hermes.config_manager import ConfigManager
from hermes.cli_hermes.banner import Banner
from hermes.cli_hermes.skin_engine import SkinEngine
from hermes.cli_hermes.profiles import ProfileManager

__all__ = [
    "HermesCLI",
    "HermesTUI",
    "ConfigManager",
    "Banner",
    "SkinEngine",
    "ProfileManager",
    "__version__",
]
