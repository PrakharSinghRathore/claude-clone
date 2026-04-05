# Claude Clone

A fully working Python clone of **Claude Code** (agentic CLI coding assistant) and **Cowork** (desktop automation tool) — built with pure Python.

## Features

### 🖥️ Two Interface Modes

- **CLI Mode** (`python main.py --cli`): A full terminal UI using `prompt_toolkit` + `rich` with streaming responses, syntax highlighting, file autocomplete, and slash commands
- **GUI Mode** (`python main.py`): A desktop application using `tkinter` with a file tree sidebar, task checklist, chat area, and settings dialog

### 🤖 Agentic AI Assistant

- **Full agentic loop**: The agent thinks, acts, observes, and iterates — up to 10 rounds per message
- **Streaming responses**: Every token streams in real-time
- **25+ built-in tools**: File operations, code search, shell execution, web search, git integration, linting, formatting, and more
- **Multi-turn conversation**: Full context preserved across messages
- **Smart context injection**: Automatically includes CWD, OS info, git status, project type detection

### 🛠️ Built-in Tools

| Category | Tools |
|----------|-------|
| File | `read_file`, `write_file`, `edit_file`, `append_file`, `delete_file`, `move_file`, `copy_file` |
| Directory | `list_directory`, `create_directory`, `get_project_structure` |
| Search | `search_files`, `grep`, `find_definition` |
| Execution | `run_command`, `run_python`, `run_script` |
| Web | `web_search` (DuckDuckGo), `fetch_url` |
| Code | `lint_python` (ruff/flake8), `format_python` (black), `get_git_status`, `git_diff`, `git_log` |
| System | `get_environment`, `install_package`, `which` |

### 📡 MCP Client

- Connect to MCP servers via **stdio** (subprocess) or **SSE** (HTTP)
- Load server configs from `~/.claude.json`
- Auto-merge MCP tools into the agent's tool registry

### ⌨️ CLI Features

- Multi-line input (`Shift+Enter`)
- File path autocomplete (`@` trigger)
- Command history (Up/Down arrows)
- Slash commands: `/clear`, `/context`, `/model`, `/tools`, `/compact`, `/export`, `/help`, `/vim`, `/cost`, `/env`, `/git`
- Vim keybinding mode (`/vim`)
- Token count and cost estimates after every response
- Syntax-highlighted markdown rendering

### 🖼️ GUI Features

- Live file tree with file watching (watchdog)
- Task checklist with status tracking
- Quick action buttons: Explain Code, Fix Bugs, Write Tests, Refactor
- Collapsible tool call cards with input/output
- Settings dialog: API key, model selection, theme
- Save/load conversations (JSON)
- Export conversations to Markdown
- Resizable panels with dark/light themes

## Quick Start

### 1. Install Dependencies

```bash
cd claude_clone
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Or create `~/.claude_clone/config.json`:
```json
{
  "api_key": "sk-ant-your-key-here",
  "model": "claude-sonnet-4-20250514"
}
```

### 3. Run

```bash
# Launch the desktop GUI
python main.py

# Launch the terminal CLI
python main.py --cli

# CLI with vim keybindings
python main.py --cli --vim
```

## Command Line Options

```
python main.py [OPTIONS]

Options:
  --cli              Launch CLI mode (Claude Code terminal)
  --vim              Enable vim keybindings (CLI mode)
  --model MODEL      AI model to use
  --theme THEME      Color theme: dark or light
  --max-tokens N     Max tokens per response (default: 8192)
  --max-iterations N Max agent iterations (default: 10)
  --cwd PATH         Set working directory
  --version          Show version info
```

## Configuration

Configuration is stored in `~/.claude_clone/config.json`:

```json
{
  "api_key": "sk-ant-...",
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 8192,
  "max_iterations": 10,
  "temperature": 1.0,
  "theme": "dark",
  "mcp_servers": [],
  "allowed_tools": [],
  "disabled_tools": [],
  "context_files": []
}
```

You can also use a `.env` file in the project directory:

```
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-20250514
```

## Project Structure

```
claude_clone/
├── main.py                  # Entry point (CLI or GUI mode)
├── config.py                # API key, model, settings management
├── requirements.txt
├── agent/
│   ├── __init__.py
│   ├── core.py              # Agentic loop with streaming events
│   ├── tools.py             # All 25+ tool implementations
│   └── mcp.py               # MCP server client (stdio + SSE)
├── cli/
│   ├── __init__.py
│   ├── app.py               # Claude Code CLI (prompt_toolkit + rich)
│   └── renderer.py          # Markdown + syntax highlighting renderer
└── gui/
    ├── __init__.py
    ├── app.py               # Cowork desktop GUI (tkinter)
    ├── sidebar.py           # File tree + task history sidebar
    └── widgets.py           # Custom tkinter widgets
```

## Tech Stack

| Package | Purpose |
|---------|---------|
| `anthropic` | Anthropic SDK for streaming + tool use |
| `rich` | Terminal UI, markdown rendering, syntax highlighting |
| `prompt_toolkit` | CLI input, autocomplete, keybindings, history |
| `tkinter` | Desktop GUI (stdlib) |
| `httpx` | Async HTTP for web fetch + MCP SSE |
| `watchdog` | File system watching (GUI sidebar) |
| `pathspec` | .gitignore-style file filtering |
| `python-dotenv` | .env file loading |
| `chardet` | File encoding detection |

## License

MIT
