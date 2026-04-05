# Claude Clone v1.2.0

> A fully working Python clone of **Claude Code** (agentic CLI coding assistant) and **Cowork** (desktop automation tool) — built with pure Python. 50 files, 48,000+ lines of code, 61 tools, 20 specialist teams, and a self-improving system.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  - [Dual Interface Modes](#-dual-interface-modes)
  - [Agentic AI Loop](#-agentic-ai-loop)
  - [61 Built-in Tools](#-61-built-in-tools)
  - [20 Specialist Teams](#-20-specialist-teams)
  - [Self-Improving System](#-self-improving-system)
  - [Desktop Automation](#-desktop-automation)
  - [Memory & Context](#-memory--context)
  - [Code Analysis](#-code-analysis)
  - [Security Scanner](#-security-scanner)
  - [Sandboxed Execution](#-sandboxed-execution)
  - [One-Click Deployment](#-one-click-deployment)
  - [MCP Client](#-mcp-client)
  - [Plugin System](#-plugin-system)
- [Quick Start](#quick-start)
- [Command Line Options](#command-line-options)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Claude Clone is a feature-rich AI coding assistant that replicates and extends the capabilities of Anthropic's Claude Code and Cowork products. It runs entirely in Python, supports both CLI and GUI interfaces, and connects to AI models via OpenRouter or the Anthropic API. The agent uses an agentic **Think → Act → Observe → Iterate** loop to autonomously complete complex multi-step coding tasks, while 20 specialist teams and 61 tools give it deep capabilities across every area of software development.

**What makes it special:**
- **Self-improving** — the AI analyzes its own code, patches bugs, extends capabilities, and optimizes performance
- **20 specialist teams** — dedicated agents for search, codegen, debugging, review, testing, security, deployment, and more
- **Desktop automation** — clipboard monitoring, window management, mouse/keyboard control, voice input, and OCR
- **Full-stack toolset** — 61 tools covering files, search, execution, web, git, databases, security, deployment, and memory
- **Dual interface** — terminal CLI (Claude Code style) and desktop GUI (Cowork style)

---

## Features

### 🖥️ Dual Interface Modes

- **CLI Mode** (`python main.py --cli`): A full terminal UI using `prompt_toolkit` + `rich` with streaming responses, syntax highlighting, file autocomplete, and slash commands
- **GUI Mode** (`python main.py`): A desktop application using `tkinter` with a file tree sidebar, task checklist, chat area, and settings dialog

### 🤖 Agentic AI Loop

The core engine uses a full agentic loop — the AI **thinks**, **acts** (calls tools), **observes** results, and **iterates** — up to 10 rounds per message. This allows it to autonomously break down complex tasks, read files, run commands, debug errors, and refine its approach until the task is complete. Streaming responses deliver every token in real-time.

### 🛠️ 61 Built-in Tools

All tools are async Python functions with auto-generated JSON schemas. Here's the full breakdown:

| Category | Tools | Description |
|----------|-------|-------------|
| **File** | `read_file`, `write_file`, `edit_file`, `append_file`, `delete_file`, `move_file`, `copy_file` | Full file CRUD with encoding detection, path safety checks |
| **Directory** | `list_directory`, `create_directory`, `get_project_structure` | Tree view, hidden file toggle, depth control |
| **Search** | `search_files`, `grep`, `find_definition` | Glob search, regex grep, AST-based symbol finding |
| **Execution** | `run_command`, `run_python`, `run_script` | Shell execution, Python eval, script runner |
| **Web** | `web_search`, `fetch_url` | DuckDuckGo search, HTML content extraction |
| **Code Quality** | `lint_python`, `format_python` | Ruff/flake8 linting, Black formatting |
| **Git** | `get_git_status`, `git_diff`, `git_log`, `git_smart_commit`, `git_repo_stats` | Full git integration with smart commit messages |
| **System** | `get_environment`, `install_package`, `which` | System info, pip install, command locator |
| **Sandbox** | `sandbox_execute`, `sandbox_install_package`, `sandbox_list_files` | Isolated code execution with memory/time limits |
| **Memory** | `memory_search`, `memory_save`, `memory_list_sessions`, `memory_export` | Persistent SQLite-backed conversation memory |
| **Analysis** | `analyze_project`, `analyze_complexity`, `analyze_dependencies`, `analyze_dead_code` | Deep static analysis: complexity scoring, dependency graphs, dead code detection |
| **Security** | `security_scan`, `security_scan_secrets`, `security_scan_dependencies` | Vulnerability scanning, secret detection, dependency audit |
| **Deployment** | `deploy_project`, `detect_deploy_platform` | One-click deploy to Docker, Railway, Vercel, AWS, GCP, Azure |
| **Database** | `db_query`, `db_list_tables` | Direct SQL query execution and schema inspection |
| **Desktop** | `desktop_screenshot`, `desktop_clipboard`, `desktop_mouse_click`, `desktop_type_text`, `desktop_hotkey`, `desktop_launch_app`, `desktop_open_url`, `desktop_speak`, `desktop_list_windows`, `desktop_focus_window`, `desktop_close_window`, `desktop_system_info`, `desktop_processes`, `desktop_active_window` | Full desktop control: screen capture, OCR, mouse, keyboard, window management, voice |
| **Self-Improve** | `self_improve_scan`, `self_improve_run`, `self_improve_status`, `self_improve_report`, `self_improve_feedback` | Self-analysis, auto-patching, optimization, and user feedback loop |

### 👥 20 Specialist Teams

The agent can delegate tasks to specialized sub-agents, each with their own system prompts and tool sets:

| Team | Purpose |
|------|---------|
| `search` | Deep code search and information retrieval |
| `codegen` | Code generation and scaffolding |
| `debug` | Bug investigation and root cause analysis |
| `review` | Code review and quality assessment |
| `test` | Test generation and execution |
| `refactor` | Code restructuring and optimization |
| `docs` | Documentation generation |
| `security` | Security auditing and vulnerability assessment |
| `perf` | Performance profiling and optimization |
| `devops` | CI/CD, infrastructure, and operations |
| `database` | Database design, queries, and migration |
| `api` | API design and implementation |
| `frontend` | UI/UX and frontend development |
| `backend` | Server-side architecture and development |
| `data` | Data processing and analysis |
| `architect` | System design and architecture planning |
| `git` | Version control operations |
| `requirements` | Requirements analysis and gathering |
| `deploy` | Deployment and release management |
| `learn` | Learning new technologies and patterns |

### 🧬 Self-Improving System

The AI that improves itself. Enabled with `--self-improve`, this system has 7 coordinated subsystems:

| Subsystem | File | What It Does |
|-----------|------|-------------|
| **Safety** | `safety.py` | Guardrails, approval gates, backup/rollback, quarantine zone, protected files |
| **Evaluator** | `evaluator.py` | Deep static analysis, code quality scoring, bug detection, function/class metrics |
| **Patcher** | `patcher.py` | Automatic bug fix generation with verified application |
| **Extender** | `extender.py` | Generates new tools to fill capability gaps |
| **Optimizer** | `optimizer.py` | Performance profiling and bottleneck optimization |
| **Learner** | `learner.py` | User preference learning and behavior adaptation |
| **Evolution** | `evolution.py` | Timeline tracking, improvement metrics, generational lineage |

All changes go through a safety review gate — the system won't modify protected files, respects change size limits, and maintains a full rollback history.

### 🖥️ Desktop Automation

Full desktop control through the `agent/desktop/` module:

- **Awareness**: Monitor clipboard changes, track active windows, capture screenshots with OCR
- **Controller**: Smooth mouse movement, human-like typing patterns, keyboard hotkeys
- **Voice**: Speech-to-text input (Google/WSR), text-to-speech output (pyttsx3), configurable wake word
- **Permissions**: Granular permission levels (STANDARD/EXPERT/TRUSTED), audit logging, auto-approve for reads

### 🧠 Memory & Context

- Persistent SQLite-backed memory database (`~/.claude_clone/memory.db`)
- Automatic conversation summarization to stay within context limits
- Cross-session memory: recall previous conversations and decisions
- Smart context injection: CWD, OS info, git status, project type detection
- 90-day retention with configurable limits

### 🔍 Code Analysis

Deep static analysis engine (`agent/analyzer.py`):

- Cyclomatic complexity scoring per function
- Dependency graph generation (uses NetworkX)
- Dead code detection across the project
- Code quality metrics and trends
- Snapshot comparisons between analysis runs

### 🛡️ Security Scanner

Built-in security scanning (`agent/security.py`):

- Vulnerability detection (SQL injection, XSS, hardcoded secrets, etc.)
- Secret scanning across codebase (API keys, passwords, tokens)
- Dependency vulnerability audit
- Configurable severity thresholds
- `.claudescanignore` support for false positives

### 🔒 Sandboxed Execution

Safe code execution environment (`agent/sandbox.py`):

- Memory-limited execution (configurable, default 512MB)
- Timeout enforcement (configurable, default 30s)
- Support for Python, JavaScript, and Bash
- Auto-cleanup of sandbox resources
- Package installation within sandbox

### 🚀 One-Click Deployment

Deploy directly from the CLI (`agent/deployer.py`):

- **Docker**: Auto-generates Dockerfile, builds, and pushes
- **Railway**: One-command deploy with auto-detection
- **Vercel**: Frontend/static site deployment
- **AWS / GCP / Azure**: Cloud platform deployment
- Platform auto-detection from project structure
- Deployment history tracking (last 50 deploys)
- Health check monitoring after deploy

### 📡 MCP Client

Connect to external MCP (Model Context Protocol) servers:

- **stdio** transport (subprocess-based)
- **SSE** transport (HTTP-based)
- Load server configs from `~/.claude.json`
- Auto-merge MCP tools into the agent's tool registry

### 🧩 Plugin System

Hot-reloadable plugin architecture (`plugins/loader.py`):

- Load plugins from `~/.claude_clone/plugins/`
- Auto-reload on file changes (configurable interval)
- Plugins can register custom tools and commands

### ⌨️ CLI Features

- Multi-line input (`Shift+Enter`)
- File path autocomplete (`@` trigger)
- Command history (Up/Down arrows)
- Slash commands: `/clear`, `/context`, `/model`, `/tools`, `/compact`, `/export`, `/help`, `/vim`, `/cost`, `/env`, `/git`
- Vim keybinding mode (`/vim` or `--vim`)
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

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/PrakharSinghRathore/claude-clone.git
cd claude-clone
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Your API Key

**Option A: OpenRouter (recommended — supports 200+ models)**

```bash
export OPENROUTER_API_KEY=sk-or-your-key-here
```

**Option B: Anthropic Direct**

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Option C: Config file** (`~/.claude_clone/config.json`)

```json
{
  "provider": "openrouter",
  "base_url": "https://openrouter.ai/api/v1",
  "model": "anthropic/claude-sonnet-4-20250514",
  "max_tokens": 8192,
  "max_iterations": 10
}
```

### 4. Run

```bash
# Launch the desktop GUI
python main.py

# Launch the terminal CLI
python main.py --cli

# CLI with vim keybindings
python main.py --cli --vim

# Enable self-improving system
python main.py --cli --self-improve

# Use a specific model
python main.py --cli --model anthropic/claude-opus-4-20250514
```

---

## Command Line Options

```
python main.py [OPTIONS]

Options:
  --cli                Launch CLI mode (Claude Code terminal)
  --vim                Enable vim keybindings (CLI mode)
  --model MODEL        AI model to use (default: anthropic/claude-sonnet-4-20250514)
  --theme THEME        Color theme: dark or light (default: dark)
  --max-tokens N       Max tokens per response (default: 8192)
  --max-iterations N   Max agent iterations per message (default: 10)
  --cwd PATH           Set working directory
  --provider PROVIDER  API provider: openrouter or anthropic (default: openrouter)
  --agent AGENT        Start with a specific specialist agent (e.g., debug, codegen, security)
  --self-improve       Enable the self-improving system
  --version            Show version info (v1.2.0)
```

---

## Configuration

Configuration is stored in `~/.claude_clone/config.json`. All sections are optional — defaults are used for anything not specified.

```json
{
  "api_key": "sk-or-...",
  "provider": "openrouter",
  "base_url": "https://openrouter.ai/api/v1",
  "model": "anthropic/claude-sonnet-4-20250514",
  "max_tokens": 8192,
  "max_iterations": 10,
  "temperature": 1.0,
  "theme": "dark",
  "active_agent": null,
  "allowed_tools": [],
  "disabled_tools": [],
  "auto_approve_tools": [],
  "mcp_servers": [],
  "context_files": [],
  "cost_warning_threshold": 1.0,

  "sandbox": {
    "enabled": true,
    "max_memory_mb": 512,
    "default_timeout": 30,
    "auto_cleanup": true,
    "allowed_languages": ["python", "javascript", "bash"]
  },
  "memory": {
    "enabled": true,
    "db_path": "~/.claude_clone/memory.db",
    "auto_summarize": true,
    "max_context_tokens": 4000,
    "retention_days": 90
  },
  "analyzer": {
    "enabled": true,
    "auto_analyze": false,
    "snapshot_on_analyze": true,
    "max_complexity_threshold": 15,
    "min_quality_score": 60
  },
  "security": {
    "enabled": true,
    "auto_scan": false,
    "severity_threshold": "MEDIUM",
    "ignore_file": ".claudescanignore",
    "scan_on_save": false
  },
  "deployment": {
    "default_platform": "docker",
    "history_limit": 50,
    "auto_health_check": true,
    "health_check_timeout": 30
  },
  "desktop": {
    "enabled": true,
    "mode": "ACTIVE",
    "awareness": {
      "monitor_clipboard": true,
      "monitor_windows": true,
      "screenshot_on_request": true,
      "ocr_enabled": true
    },
    "voice": {
      "enabled": false,
      "stt_engine": "google",
      "tts_engine": "pyttsx3",
      "wake_word": "hey claude",
      "language": "en-US"
    },
    "permissions": {
      "level": "STANDARD",
      "auto_approve_read": true,
      "ask_before_delete": true,
      "audit_log": true
    }
  },
  "self_improving": {
    "enabled": false,
    "auto_improve": false,
    "improvement_interval": 3600,
    "max_patches_per_cycle": 10,
    "max_extensions_per_cycle": 3,
    "max_optimizations_per_cycle": 5
  }
}
```

You can also use a `.env` file in the project directory:

```env
OPENROUTER_API_KEY=sk-or-...
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=anthropic/claude-sonnet-4-20250514
CLAUDE_MAX_TOKENS=8192
API_PROVIDER=openrouter
```

---

## Project Structure

```
claude_clone/
├── main.py                          # Entry point (CLI or GUI mode)
├── config.py                        # Full configuration management
├── requirements.txt                 # Python dependencies (25 packages)
│
├── agent/                           # Core AI agent
│   ├── __init__.py                  # Agent exports
│   ├── core.py                      # Agentic loop: Think → Act → Observe → Iterate
│   ├── tools.py                     # All 61 tool implementations
│   ├── teams.py                     # 20 specialist team definitions
│   ├── mcp.py                       # MCP server client (stdio + SSE)
│   ├── memory.py                    # SQLite-backed persistent memory
│   ├── sandbox.py                   # Sandboxed code execution
│   ├── security.py                  # Security & vulnerability scanner
│   ├── analyzer.py                  # Code analysis engine
│   ├── deployer.py                  # One-click deployment
│   ├── model_router.py              # Multi-model routing & fallback
│   ├── plan_mode.py                 # Task planning & decomposition
│   ├── task_queue.py                # Background task management
│   ├── session_recorder.py          # Session recording & playback
│   ├── diff_preview.py              # Git-style diff visualization
│   ├── feedback.py                  # User feedback collection
│   ├── evaluator.py                 # Response quality evaluation
│   ├── indexer.py                   # Project indexing engine
│   ├── mentions.py                  # @mention handling in conversations
│   ├── self_improving.py            # Self-improving system bridge
│   │
│   ├── self_improving/              # Self-Improving System (7 modules)
│   │   ├── __init__.py              # Orchestrator & exports
│   │   ├── safety.py                # Guardrails, approval gates, rollback
│   │   ├── evaluator.py             # Deep static analysis & scoring
│   │   ├── patcher.py               # Bug fix generation & application
│   │   ├── extender.py              # New tool generation
│   │   ├── optimizer.py             # Performance profiling & optimization
│   │   ├── learner.py               # User preference learning
│   │   └── evolution.py             # Timeline tracking & lineage
│   │
│   └── desktop/                     # Desktop Automation
│       ├── __init__.py              # Desktop module init
│       ├── orchestrator.py          # Desktop orchestration
│       ├── awareness.py             # Clipboard, windows, screenshots, OCR
│       ├── controller.py            # Mouse, keyboard, hotkey control
│       ├── permissions.py           # Permission system & audit log
│       └── voice.py                 # Speech-to-text & text-to-speech
│
├── cli/                             # Claude Code Terminal Interface
│   ├── __init__.py
│   ├── app.py                       # CLI application (prompt_toolkit + rich)
│   └── renderer.py                  # Markdown + syntax highlighting renderer
│
├── gui/                             # Cowork Desktop Interface
│   ├── __init__.py
│   ├── app.py                       # Desktop GUI application (tkinter)
│   ├── sidebar.py                   # File tree + task history sidebar
│   └── widgets.py                   # Custom tkinter widgets
│
├── plugins/                         # Plugin System
│   ├── __init__.py
│   └── loader.py                    # Hot-reloadable plugin loader
│
└── utils/                           # Shared Utilities
    ├── __init__.py
    ├── code_diff.py                 # Unified diff generation
    ├── database.py                  # Database helpers
    ├── git_manager.py               # Git operations wrapper
    └── webhook.py                   # Webhook integration
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Entry Point)                    │
│                   --cli  ──→  CLI Mode (prompt_toolkit)         │
│                   default ──→  GUI Mode (tkinter)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      agent/core.py (Agent)                       │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              Agentic Loop (max 10 iterations)            │  │
│   │                                                          │  │
│   │    ┌─────────┐    ┌─────────┐    ┌──────────────────┐   │  │
│   │    │  THINK   │───→│  ACT    │───→│   OBSERVE        │   │  │
│   │    │ (reason) │    │ (tools) │    │ (read results)   │   │  │
│   │    └─────────┘    └─────────┘    └────────┬─────────┘   │  │
│   │         ▲                                   │             │  │
│   │         └───────────────────────────────────┘             │  │
│   │                      ITERATE                              │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│   │   Teams     │  │   Memory     │  │  Self-Improving    │   │
│   │  (20 agents)│  │  (SQLite)    │  │  (7 subsystems)    │   │
│   └─────────────┘  └──────────────┘  └────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    agent/tools.py (61 Tools)                     │
│                                                                  │
│   Files │ Search │ Exec │ Web │ Git │ DB │ Desktop │ Security   │
│   Sandbox │ Memory │ Analysis │ Deploy │ Self-Improve            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     API Provider Layer                           │
│                                                                  │
│   ┌──────────────────┐    ┌──────────────────────┐             │
│   │   OpenRouter     │ or │   Anthropic Direct   │             │
│   │ (200+ models)    │    │  (Claude models)     │             │
│   └──────────────────┘    └──────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Package | Purpose |
|---------|---------|
| `anthropic` | Anthropic SDK for streaming + tool use |
| `rich` | Terminal UI, markdown rendering, syntax highlighting |
| `prompt_toolkit` | CLI input, autocomplete, keybindings, history |
| `tkinter` | Desktop GUI (stdlib) |
| `httpx` | Async HTTP for web fetch + MCP SSE |
| `aiohttp` | Async HTTP client for MCP and webhooks |
| `watchdog` | File system watching (GUI sidebar) |
| `pathspec` | .gitignore-style file filtering |
| `python-dotenv` | .env file loading |
| `chardet` | File encoding detection |
| `websockets` | WebSocket support for MCP and collaboration |
| `networkx` | Dependency graph analysis |
| `psutil` | System resource monitoring |
| `pyautogui` | Desktop mouse/keyboard automation |
| `Pillow` | Image processing for screenshots |
| `pytesseract` | OCR for screenshot text extraction |
| `pyttsx3` | Text-to-speech for desktop voice |
| `SpeechRecognition` | Speech-to-text for voice input |
| `psycopg2-binary` | PostgreSQL database support |
| `cryptography` | Secure credential storage |

---

## License

MIT
