"""
Interactive setup wizard for Hermes CLI first-run configuration.

Guides users through:
- API key setup (provider selection, key entry, validation)
- Model selection and testing
- Memory backend configuration
- Platform adapter setup
- Theme and preference selection
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes.cli_hermes.config_manager import ConfigManager, DEFAULT_CONFIG_DIR


# ──────────────────────────────────────────────
# Setup Wizard Steps
# ──────────────────────────────────────────────

WELCOME_TEXT = """
    Welcome to Hermes CLI!

    This wizard will guide you through the initial setup.
    You can always change these settings later with /config or /setup.

    Press Enter to accept defaults shown in [brackets].
"""

KNOWN_PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "description": "Multi-provider gateway with many models",
        "base_url": "https://openrouter.ai/api/v1",
        "env_var": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/keys",
        "models": [
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-opus-4-20250514",
            "anthropic/claude-3-5-haiku-20241022",
            "google/gemini-2.5-pro-preview",
            "meta-llama/llama-4-maverick",
            "openai/gpt-4o",
        ],
    },
    "anthropic": {
        "name": "Anthropic Direct",
        "description": "Direct Anthropic API access",
        "base_url": "https://api.anthropic.com",
        "env_var": "ANTHROPIC_API_KEY",
        "key_url": "https://console.anthropic.com/account/keys",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514",
            "claude-3-5-haiku-20241022",
        ],
    },
}

KNOWN_THEMES = [
    ("dark", "Default dark theme"),
    ("light", "Clean light theme"),
    ("nord", "Arctic blue palette"),
    ("dracula", "Dark purple theme"),
    ("solarized", "Solarized color scheme"),
    ("catppuccin", "Soothing dark theme"),
    ("monokai", "Classic Monokai"),
    ("gruvbox", "Warm retro theme"),
]

PROMPT_STYLES = [
    ("hermes", "Default Hermes style (>>>)"),
    ("claude", "Claude-style prompt"),
    ("minimal", "Minimal prompt (>)"),
    ("powerline", "Powerline-style arrows"),
    ("starship", "Starship-inspired prompt"),
    ("fancy", "Fancy diamond prompt"),
]


class SetupWizard:
    """Interactive first-run setup wizard."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.config.load()
        self._input_fn = input
        self._print_fn = print
        self._completed_steps: List[str] = []

    def run(self, skip_api: bool = False) -> Dict[str, Any]:
        """Run the full setup wizard."""
        self._print_fn(WELCOME_TEXT)

        results = {}

        if not skip_api:
            results.update(self._step_api_provider())
            results.update(self._step_api_key())

        results.update(self._step_model_selection())
        results.update(self._step_theme_selection())
        results.update(self._step_prompt_style())
        results.update(self._step_preferences())
        results.update(self._step_features())

        # Save configuration
        self._save_config(results)

        self._print_fn("\n  \033[32mSetup complete!\033[0m")
        self._print_fn("  You can start using Hermes CLI now.")
        self._print_fn("  Type /help to see all available commands.\n")

        return results

    def _step_api_provider(self) -> Dict[str, Any]:
        """Step 1: Select API provider."""
        self._print_step_header("API Provider")

        # Check for existing keys
        existing = []
        for provider_id, info in KNOWN_PROVIDERS.items():
            if os.environ.get(info["env_var"]):
                existing.append(provider_id)

        if existing:
            self._print_fn(f"  \033[32mFound API keys for: {', '.join(existing)}\033[0m")

        self._print_fn("\n  Available providers:")
        providers = list(KNOWN_PROVIDERS.keys())
        for i, pid in enumerate(providers, 1):
            info = KNOWN_PROVIDERS[pid]
            default_marker = " (default)" if pid == "openrouter" else ""
            self._print_fn(f"    {i}. {info['name']}{default_marker}")
            self._print_fn(f"       {info['description']}")

        while True:
            choice = self._prompt("  Select provider", "1")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(providers):
                    selected = providers[idx]
                    break
            except (ValueError, IndexError):
                pass
            self._print_fn("  \033[31mInvalid choice. Please enter a number.\033[0m")

        provider_info = KNOWN_PROVIDERS[selected]
        self._print_fn(f"  Selected: {provider_info['name']}\n")
        self._completed_steps.append("provider")

        return {
            "provider": selected,
            "base_url": provider_info["base_url"],
        }

    def _step_api_key(self) -> Dict[str, Any]:
        """Step 2: Enter API key."""
        self._print_step_header("API Key")

        provider = self.config.get("provider", "openrouter")
        provider_info = KNOWN_PROVIDERS.get(provider, KNOWN_PROVIDERS["openrouter"])

        # Check for existing key
        existing_key = os.environ.get(provider_info["env_var"], "")
        if existing_key:
            self._print_fn(f"  \033[32mAPI key found in {provider_info['env_var']}\033[0m")
            use_existing = self._prompt("  Use existing key?", "y")
            if use_existing.lower() in ("y", "yes", ""):
                self._print_fn("  Using existing API key.\n")
                self._completed_steps.append("api_key")
                return {"api_key": existing_key}

        self._print_fn(f"  Get your API key from: {provider_info['key_url']}")
        self._print_fn("  (The key will be stored securely in your config directory)\n")

        while True:
            key = self._prompt("  Enter API key", "")
            if not key:
                self._print_fn("  \033[33mNo API key entered. You can set it later.\033[0m")
                break
            if len(key) < 10:
                self._print_fn("  \033[31mAPI key seems too short. Please try again.\033[0m")
                continue
            break

        self._print_fn("")
        self._completed_steps.append("api_key")
        return {"api_key": key}

    def _step_model_selection(self) -> Dict[str, Any]:
        """Step 3: Select default model."""
        self._print_step_header("Model Selection")

        provider = self.config.get("provider", "openrouter")
        provider_info = KNOWN_PROVIDERS.get(provider, KNOWN_PROVIDERS["openrouter"])
        models = provider_info.get("models", ["anthropic/claude-sonnet-4-20250514"])

        self._print_fn("  Available models:")
        default_model = "anthropic/claude-sonnet-4-20250514"
        for i, model in enumerate(models, 1):
            is_default = " (recommended)" if model == default_model else ""
            self._print_fn(f"    {i}. {model}{is_default}")

        self._print_fn(f"    {len(models) + 1}. Enter custom model name")

        while True:
            choice = self._prompt("  Select model", "1")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected_model = models[idx]
                    break
                elif idx == len(models):
                    selected_model = self._prompt("  Enter model name", default_model)
                    if selected_model:
                        break
            except (ValueError, IndexError):
                pass

        # Test model connectivity
        test = self._prompt("  Test model connectivity?", "y")
        if test.lower() in ("y", "yes", ""):
            self._test_model(selected_model)

        self._print_fn(f"  Selected model: {selected_model}\n")
        self._completed_steps.append("model")
        return {"model": selected_model}

    def _step_theme_selection(self) -> Dict[str, Any]:
        """Step 4: Select visual theme."""
        self._print_step_header("Theme Selection")

        self._print_fn("  Available themes:")
        for i, (name, desc) in enumerate(KNOWN_THEMES, 1):
            self._print_fn(f"    {i}. {name} - {desc}")

        while True:
            choice = self._prompt("  Select theme", "1")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(KNOWN_THEMES):
                    selected_theme = KNOWN_THEMES[idx][0]
                    break
            except (ValueError, IndexError):
                pass
            self._print_fn("  \033[31mInvalid choice.\033[0m")

        self._print_fn(f"  Selected theme: {selected_theme}\n")
        self._completed_steps.append("theme")
        return {"theme": selected_theme, "skin": selected_theme}

    def _step_prompt_style(self) -> Dict[str, Any]:
        """Step 5: Select prompt style."""
        self._print_step_header("Prompt Style")

        self._print_fn("  Available prompt styles:")
        for i, (name, desc) in enumerate(PROMPT_STYLES, 1):
            self._print_fn(f"    {i}. {name} - {desc}")

        while True:
            choice = self._prompt("  Select prompt style", "1")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(PROMPT_STYLES):
                    selected_style = PROMPT_STYLES[idx][0]
                    break
            except (ValueError, IndexError):
                pass

        self._print_fn(f"  Selected prompt style: {selected_style}\n")
        self._completed_steps.append("prompt_style")
        return {"prompt_style": selected_style}

    def _step_preferences(self) -> Dict[str, Any]:
        """Step 6: General preferences."""
        self._print_step_header("Preferences")

        prefs = {}

        # Sound
        sound = self._prompt("  Enable sound effects?", "n")
        prefs["sound_enabled"] = sound.lower() in ("y", "yes")

        # Notifications
        notif = self._prompt("  Enable desktop notifications?", "n")
        prefs["notification_enabled"] = notif.lower() in ("y", "yes")

        # Auto-save
        auto_save = self._prompt("  Auto-save sessions?", "y")
        prefs["auto_save"] = auto_save.lower() in ("y", "yes", "")

        # Streaming
        streaming = self._prompt("  Enable streaming output?", "y")
        prefs["streaming"] = streaming.lower() in ("y", "yes", "")

        # Markdown rendering
        md_render = self._prompt("  Enable markdown rendering?", "y")
        prefs["markdown_render"] = md_render.lower() in ("y", "yes", "")

        # Syntax highlighting
        syntax = self._prompt("  Enable syntax highlighting?", "y")
        prefs["syntax_highlight"] = syntax.lower() in ("y", "yes", "")

        # Emoji
        emoji = self._prompt("  Enable emoji in output?", "y")
        prefs["emoji_enabled"] = emoji.lower() in ("y", "yes", "")

        self._completed_steps.append("preferences")
        return prefs

    def _step_features(self) -> Dict[str, Any]:
        """Step 7: Feature enable/disable."""
        self._print_step_header("Features")

        features = {}

        # Memory
        memory = self._prompt("  Enable memory (conversation history)?", "y")
        features["memory_enabled"] = memory.lower() in ("y", "yes", "")

        # Gateway
        gateway = self._prompt("  Enable gateway (multi-platform)?", "n")
        features["gateway_enabled"] = gateway.lower() in ("y", "yes")

        # Cron
        cron = self._prompt("  Enable cron jobs?", "n")
        features["cron_enabled"] = cron.lower() in ("y", "yes")

        # Self-improving
        self_imp = self._prompt("  Enable self-improving system?", "n")
        features["self_improving_enabled"] = self_imp.lower() in ("y", "yes")

        self._completed_steps.append("features")
        return features

    def _save_config(self, results: Dict[str, Any]) -> None:
        """Save all configuration from wizard results."""
        for key, value in results.items():
            if value is not None:
                self.config.set(key, value)

        # Handle nested features
        if results.get("gateway_enabled") is not None:
            self.config.set("gateway.enabled", results["gateway_enabled"])
        if results.get("cron_enabled") is not None:
            self.config.set("cron.enabled", results["cron_enabled"])

        self.config.save()

    def _test_model(self, model: str) -> bool:
        """Test model connectivity with a simple request."""
        self._print_fn("  Testing model connectivity...")

        api_key = self.config.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            self._print_fn("  \033[33mSkipped: No API key available.\033[0m")
            return False

        try:
            import httpx

            provider = self.config.get("provider", "openrouter")
            base_url = self.config.get("base_url", "https://openrouter.ai/api/v1")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "Say 'Hello!' in one word."}],
                "max_tokens": 10,
            }

            # Use synchronous httpx for the setup wizard
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

            if resp.status_code == 200:
                data = resp.json()
                message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                self._print_fn(f"  \033[32mConnected! Response: {message.strip()}\033[0m")
                return True
            else:
                self._print_fn(f"  \033[33mConnection test returned {resp.status_code}\033[0m")
                return False

        except ImportError:
            self._print_fn("  \033[33mSkipped: httpx not installed.\033[0m")
            return False
        except Exception as e:
            self._print_fn(f"  \033[33mConnection test failed: {e}\033[0m")
            return False

    def _print_step_header(self, title: str) -> None:
        """Print a step header."""
        self._print_fn(f"\n  \033[1m── {title} ──\033[0m\n")

    def _prompt(self, prompt: str, default: str = "") -> str:
        """Prompt user for input with default."""
        if default:
            full_prompt = f"{prompt} [\033[36m{default}\033[0m]: "
        else:
            full_prompt = f"{prompt}: "
        try:
            result = self._input_fn(full_prompt).strip()
            return result if result else default
        except (EOFError, KeyboardInterrupt):
            return default

    def set_input_fn(self, fn) -> None:
        """Set custom input function (for testing)."""
        self._input_fn = fn

    def set_print_fn(self, fn) -> None:
        """Set custom print function (for testing)."""
        self._print_fn = fn
