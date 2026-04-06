# Claude Clone v3.0.0 — AI Agent Framework

> A comprehensive AI agent framework inspired by Claude Code, enhanced with multi-agent orchestration (Crew), workflow automation (Flow), an event-driven architecture, a full Atlas Agent sub-system, and 24+ messaging platform integrations. 299 Python files, 157,986 lines of code.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Module Descriptions](#module-descriptions)
- [Features](#features)
- [Quick Start](#quick-start)
- [Command Line Options](#command-line-options)
- [Project Statistics](#project-statistics)
- [License](#license)

---

## Overview

Claude Clone is a feature-rich, extensible AI agent framework that replicates and significantly extends the capabilities of Anthropic's Claude Code and Cowork products. Built entirely in Python, it provides:

- **Multi-agent orchestration** via the Crew system — delegate tasks to specialized agent teams with coordinated execution, guardrails, and training
- **Declarative workflow automation** via the Flow system — define complex multi-step pipelines using decorators (`@start`, `@listen`, `@router`)
- **Event-driven architecture** — a central event bus with typed events for cross-component communication (Agent, Crew, Task, Tool, LLM, Memory, Flow, System events)
- **Atlas Agent sub-system** — 24+ integrated subsystems including context compression, smart routing, skills, plugin SDK, canvas UI, media pipeline, multi-platform gateway, security, i18n, and more
- **61+ built-in tools** covering files, search, execution, web, git, databases, security, deployment, memory, and desktop automation
- **20 specialist teams** — dedicated agents for search, codegen, debugging, review, testing, security, deployment, and more
- **Self-improving system** — the AI analyzes its own code, patches bugs, extends capabilities, and optimizes performance through 7 coordinated subsystems
- **Knowledge base** — persistent knowledge storage with graph-based search, extraction, and import capabilities
- **Multi-provider LLM abstraction** — supports Anthropic, OpenAI, Azure, Google Gemini, Ollama, and any OpenAI-compatible API
- **Plugin system** — hot-reloadable plugins with SDK for third-party extensions
- **Dual interface** — terminal CLI (Claude Code style) and desktop GUI (Cowork style), plus a full Atlas TUI

---

## Project Structure

```
claude_clone/
├── main.py                              # Entry point (CLI / GUI / Atlas / Doctor / Stats)
├── config.py                            # Full configuration management (25+ sections)
├── atlas_bridge.py                      # Central integration facade for all Atlas subsystems
├── requirements.txt                     # Python dependencies
│
├── agent/                               # Core AI Agent
│   ├── __init__.py
│   ├── core.py                          # Agentic loop: Think -> Act -> Observe -> Iterate
│   ├── tools.py                         # 61+ tool implementations (2911 lines)
│   ├── teams.py                         # 20 specialist team definitions
│   ├── mcp.py                           # MCP server client (stdio + SSE)
│   ├── memory.py                        # SQLite-backed persistent memory
│   ├── sandbox.py                       # Sandboxed code execution
│   ├── security.py                      # Security & vulnerability scanner
│   ├── analyzer.py                      # Code analysis engine
│   ├── deployer.py                      # One-click deployment
│   ├── model_router.py                  # Multi-model routing & fallback
│   ├── plan_mode.py                     # Task planning & decomposition
│   ├── task_queue.py                    # Background task management
│   ├── session_recorder.py              # Session recording & playback
│   ├── diff_preview.py                  # Git-style diff visualization
│   ├── feedback.py                      # User feedback collection
│   ├── evaluator.py                     # Response quality evaluation
│   ├── indexer.py                       # Project indexing engine
│   ├── mentions.py                      # @mention handling in conversations
│   ├── self_improving.py                # Self-improving system bridge
│   │
│   ├── self_improving/                  # Self-Improving System (7 modules)
│   │   ├── __init__.py
│   │   ├── safety.py                    # Guardrails, approval gates, rollback
│   │   ├── evaluator.py                 # Deep static analysis & scoring
│   │   ├── patcher.py                   # Bug fix generation & application
│   │   ├── extender.py                  # New tool generation
│   │   ├── optimizer.py                 # Performance profiling & optimization
│   │   ├── learner.py                   # User preference learning
│   │   └── evolution.py                 # Timeline tracking & lineage
│   │
│   ├── knowledge_base/                  # Persistent Knowledge Base
│   │   ├── __init__.py
│   │   ├── knowledge_store.py           # SQLite knowledge storage
│   │   ├── knowledge_graph.py           # Graph-based knowledge relationships
│   │   ├── knowledge_search.py          # Semantic and keyword search
│   │   ├── knowledge_extractor.py       # Auto-extract knowledge from conversations
│   │   └── knowledge_importer.py        # Import from Obsidian, Markdown, etc.
│   │
│   └── desktop/                         # Desktop Automation
│       ├── __init__.py
│       ├── orchestrator.py              # Desktop orchestration
│       ├── awareness.py                 # Clipboard, windows, screenshots, OCR
│       ├── controller.py                # Mouse, keyboard, hotkey control
│       ├── permissions.py               # Permission system & audit log
│       └── voice.py                     # Speech-to-text & text-to-speech
│
├── crew/                                # Multi-Agent Crew Orchestration
│   ├── __init__.py
│   ├── crew.py                          # Crew manager (coordinated multi-agent execution)
│   ├── agent.py                         # CrewAgent definition
│   ├── task.py                          # Task definition with outputs & delegation
│   ├── process.py                       # Sequential and parallel execution processes
│   ├── guardrails.py                    # Output validation & safety guardrails
│   ├── training.py                      # Iterative crew improvement through training
│   ├── cache.py                         # Response caching for performance
│   ├── rpm_controller.py                # Rate limiting per agent
│   └── usage_metrics.py                 # Token usage tracking per crew run
│
├── flow/                                # Workflow Orchestration
│   ├── __init__.py                      # Exports: Flow, start, listen, router, and_, or_
│   ├── flow.py                          # Flow engine with decorator-based step definitions
│   ├── context.py                       # FlowContext for shared state between steps
│   ├── config.py                        # FlowConfig for global flow settings
│   ├── human_feedback.py                # Human-in-the-loop feedback gates
│   ├── persistence/                     # Flow State Persistence
│   │   ├── __init__.py
│   │   ├── base.py                      # Abstract persistence backend
│   │   └── sqlite.py                    # SQLite-based flow state storage
│   └── visualization/                   # Flow Visualization
│       ├── __init__.py
│       ├── schema.py                    # Flow schema definitions
│       └── builder.py                   # Mermaid / DOT flow diagram generation
│
├── events/                              # Event System
│   ├── __init__.py                      # EventBus singleton + all event type exports
│   ├── event_bus.py                     # Publish/subscribe event bus
│   ├── base_events.py                   # BaseEvent and EventPriority
│   └── event_types/                     # Typed Events
│       ├── __init__.py
│       ├── agent_events.py              # Agent lifecycle events
│       ├── crew_events.py               # Crew execution events
│       ├── task_events.py               # Task lifecycle events
│       ├── tool_events.py               # Tool invocation events
│       ├── llm_events.py                # LLM call events (request, response, error)
│       ├── memory_events.py             # Memory store/retrieve events
│       ├── flow_events.py               # Flow step and transition events
│       └── system_events.py             # System startup, shutdown, error events
│
├── knowledge/                           # Knowledge Sources
│   ├── __init__.py                      # Exports all source types
│   ├── base.py                          # KnowledgeBase and KnowledgeSource abstractions
│   └── sources.py                       # PDF, CSV, Excel, JSON, TextFile, String sources
│
├── llm/                                 # LLM Abstraction Layer
│   ├── __init__.py                      # Exports: BaseLLM, LLMConfig, LLMProvider, LLMResponse
│   ├── base.py                          # Abstract LLM interface with config and response
│   └── provider.py                      # Multi-provider: Anthropic, OpenAI, Azure, Gemini, Ollama
│
├── hooks/                               # Lifecycle Hooks
│   ├── __init__.py                      # Exports: before/after decorators for LLM and tools
│   ├── decorators.py                    # @before_llm_call, @after_llm_call, @before_tool_call, @after_tool_call
│   └── types.py                         # HookContext and HookResult types
│
├── evaluation/                          # Agent Evaluation
│   ├── __init__.py                      # Exports: AgentEvaluator, EvaluationResult, EvaluationMetric
│   └── evaluator.py                     # Performance measurement and assessment
│
├── telemetry/                           # Telemetry & Metrics
│   ├── __init__.py                      # Exports: TelemetryTracker
│   └── tracker.py                       # Usage tracking and metrics collection
│
├── security/                            # Security Utilities
│   ├── __init__.py                      # Exports: generate_fingerprint, verify_fingerprint
│   └── fingerprint.py                   # Agent fingerprinting for identity verification
│
├── i18n/                                # Internationalization
│   ├── __init__.py                      # Exports: I18N, get_i18n
│   └── loader.py                        # Locale loader with fallback support
│
├── atlas/                               # Atlas Agent Sub-system (24 subsystems)
│   ├── __init__.py                      # Atlas package init
│   ├── constants.py                     # Shared constants
│   │
│   ├── core/                            # Atlas Core
│   │   ├── __init__.py
│   │   ├── context_compressor.py        # Context window compression (200K tokens)
│   │   ├── prompt_builder.py            # Sectioned prompt construction
│   │   ├── smart_routing.py             # Cost/latency-aware model routing
│   │   ├── memory_manager.py            # Unified memory management
│   │   ├── memory_provider.py           # Memory provider interface
│   │   ├── builtin_memory.py            # Built-in SQLite memory provider
│   │   ├── insights.py                  # Usage analytics & insights
│   │   ├── trajectory.py                # RL training data collection
│   │   ├── title_generator.py           # Auto-generate session titles
│   │   ├── display.py                   # Display utilities
│   │   ├── redact.py                    # Sensitive data redaction
│   │   ├── model_metadata.py            # Model capability metadata
│   │   ├── context_references.py        # File/URL context resolution
│   │   ├── credential_pool.py           # Multi-key credential rotation
│   │   ├── auxiliary_client.py          # Auxiliary model client
│   │   └── usage_pricing.py             # Token cost estimation
│   │
│   ├── tools/                           # Atlas Tools (30 tools)
│   │   ├── __init__.py
│   │   ├── registry.py                  # Tool registry & discovery
│   │   ├── browser_tool.py              # Headless browser automation
│   │   ├── code_execution.py            # Sandboxed code execution
│   │   ├── cronjob_tool.py              # Cron job management
│   │   ├── delegate_tool.py             # Task delegation to specialist agents
│   │   ├── file_tools.py                # Advanced file operations
│   │   ├── image_gen_tool.py            # AI image generation
│   │   ├── mcp_tool.py                  # MCP server integration
│   │   ├── memory_tool.py               # Memory store/retrieve
│   │   ├── mixture_of_agents_tool.py    # Mixture-of-Agents routing
│   │   ├── send_message_tool.py         # Cross-platform messaging
│   │   ├── session_search.py            # Session history search
│   │   ├── skill_manager.py             # Skill management
│   │   ├── skills_hub.py                # Skills discovery hub
│   │   ├── skills_tool.py               # Skill execution
│   │   ├── terminal_tool.py             # Advanced terminal control
│   │   ├── todo_tool.py                 # Todo list management
│   │   ├── transcription_tool.py        # Audio transcription
│   │   ├── tts_tool.py                  # Text-to-speech
│   │   ├── vision_tool.py               # Image understanding
│   │   ├── web_tools.py                 # Web search & fetch
│   │   └── web_search.py                # Search engine integration
│   │
│   ├── skills/                          # Skills System
│   │   ├── __init__.py
│   │   ├── manager.py                   # Skill lifecycle management
│   │   ├── loader.py                    # Skill discovery & loading
│   │   ├── executor.py                  # Skill execution engine
│   │   ├── registry.py                  # Skill registry
│   │   └── builtins/                    # Built-in Skills
│   │       ├── __init__.py
│   │       ├── code_review/             # Automated code review
│   │       ├── debug/                   # Debugging assistance
│   │       ├── deploy/                  # Deployment automation
│   │       ├── git_workflow/            # Git workflow automation
│   │       └── research/                # Research & information gathering
│   │
│   ├── channels/                        # Channel Abstractions
│   │   ├── __init__.py
│   │   ├── base.py                      # Abstract channel interface
│   │   ├── adapter.py                   # Channel adapter & routing
│   │   ├── bindings.py                  # Platform bindings
│   │   └── routing.py                   # Message routing engine
│   │
│   ├── gateway/                         # Multi-Platform Gateway
│   │   ├── __init__.py
│   │   ├── config.py                    # Gateway configuration
│   │   ├── runner.py                    # Gateway runner & lifecycle
│   │   ├── session.py                   # Gateway session management
│   │   ├── delivery.py                  # Message delivery engine
│   │   ├── hooks.py                     # Gateway lifecycle hooks
│   │   ├── mirror.py                    # Message mirroring
│   │   ├── status.py                    # Status reporting
│   │   ├── stream_consumer.py           # Stream event consumer
│   │   └── platforms/                   # 24 Platform Integrations
│   │       ├── __init__.py
│   │       ├── api_server.py            # REST API server
│   │       ├── slack.py                 # Slack
│   │       ├── discord.py               # Discord
│   │       ├── telegram.py              # Telegram
│   │       ├── whatsapp.py              # WhatsApp
│   │       ├── msteams.py               # Microsoft Teams
│   │       ├── mattermost.py            # Mattermost
│   │       ├── signal.py                # Signal
│   │       ├── irc.py                   # IRC
│   │       ├── matrix.py                # Matrix
│   │       ├── twitch.py                # Twitch
│   │       ├── email_platform.py        # Email (SMTP/IMAP)
│   │       ├── sms.py                   # SMS (Twilio)
│   │       ├── dingtalk.py              # DingTalk
│   │       ├── feishu.py                # Feishu/Lark
│   │       ├── wecom.py                 # WeCom
│   │       ├── zalo.py                  # Zalo
│   │       ├── line.py                  # LINE
│   │       ├── google_chat.py           # Google Chat
│   │       ├── nextcloud.py             # Nextcloud Talk
│   │       ├── nostr.py                 # Nostr
│   │       ├── bluebubbles.py           # BlueBubbles (iMessage)
│   │       ├── voice_call.py            # Voice call (WebRTC)
│   │       └── webhook.py               # Generic Webhook
│   │
│   ├── plugin_sdk/                      # Plugin Development Kit
│   │   ├── __init__.py
│   │   ├── core.py                      # Plugin core framework
│   │   ├── contracts.py                 # Plugin API contracts
│   │   ├── loader.py                    # Plugin loader & discovery
│   │   ├── manifest.py                  # Plugin manifest schema
│   │   ├── registry.py                  # Plugin registry
│   │   └── sandbox.py                   # Plugin sandboxing
│   │
│   ├── plugins/                         # Built-in Plugins
│   │   ├── __init__.py
│   │   └── memory/                      # Memory Provider Plugins (9 providers)
│   │       ├── __init__.py
│   │       ├── base.py                  # Memory provider base class
│   │       ├── registry.py              # Memory provider registry
│   │       ├── mem0_plugin.py           # Mem0 integration
│   │       ├── byterover.py             # ByteRover memory
│   │       ├── hindsight.py             # Hindsight memory
│   │       ├── holographic.py           # Holographic memory
│   │       ├── honcho.py                # Honcho memory
│   │       ├── openviking.py            # OpenViking memory
│   │       └── retaindb.py              # RetainDB memory
│   │
│   ├── config/                          # Atlas Configuration
│   │   ├── __init__.py
│   │   ├── loader.py                    # Configuration loading
│   │   ├── schema.py                    # JSON schema validation
│   │   └── types.py                     # Configuration type definitions
│   │
│   ├── sessions/                        # Session Management
│   │   ├── __init__.py
│   │   ├── manager.py                   # Session lifecycle manager
│   │   ├── store.py                     # Session persistence store
│   │   ├── keys.py                      # Session key management
│   │   └── transcript.py                # Session transcript handling
│   │
│   ├── security/                        # Atlas Security
│   │   ├── __init__.py
│   │   ├── policy.py                    # Security policy engine
│   │   ├── audit.py                     # Security audit logging
│   │   ├── allowlist.py                 # Path/command allowlisting
│   │   ├── sandbox.py                   # Execution sandboxing
│   │   ├── secrets.py                   # Secret management
│   │   └── pairing.py                   # Secure device pairing
│   │
│   ├── canvas/                          # Canvas / A2UI Visual Workspace
│   │   ├── __init__.py
│   │   ├── host.py                      # Canvas host server
│   │   ├── renderer.py                  # Widget rendering engine
│   │   └── push.py                      # Real-time canvas push updates
│   │
│   ├── media/                           # Media Processing Pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py                  # Media processing orchestration
│   │   ├── audio.py                     # Audio processing
│   │   ├── images.py                    # Image processing
│   │   ├── video.py                     # Video processing
│   │   ├── vision.py                    # Computer vision
│   │   └── generation.py                # Media generation (images, audio)
│   │
│   ├── acp/                             # Atlas Control Protocol (ACP)
│   │   ├── __init__.py
│   │   ├── server.py                    # ACP server (FastAPI)
│   │   ├── session.py                   # ACP session management
│   │   ├── auth.py                      # ACP authentication
│   │   ├── permissions.py               # ACP permission model
│   │   └── events.py                    # ACP event system
│   │
│   ├── cli_atlas/                       # Atlas Terminal UI
│   │   ├── __init__.py
│   │   ├── main.py                      # Atlas CLI entry point
│   │   ├── tui.py                       # Full-featured terminal UI
│   │   ├── commands.py                  # CLI command definitions
│   │   ├── callbacks.py                 # UI callback handlers
│   │   ├── banner.py                    # Startup banner
│   │   ├── doctor.py                    # Diagnostics system
│   │   ├── setup.py                     # First-run setup wizard
│   │   ├── profiles.py                  # Configuration profiles
│   │   ├── providers.py                 # Provider management
│   │   ├── config_manager.py            # In-app config management
│   │   ├── mcp_config.py                # MCP server configuration
│   │   ├── models_cmd.py                # Model selection commands
│   │   ├── gateway_cmd.py               # Gateway management commands
│   │   ├── cron_cmd.py                  # Cron job management commands
│   │   ├── tools_config.py              # Tool configuration
│   │   ├── skills_config.py             # Skill configuration
│   │   ├── skills_hub.py                # Skills discovery hub
│   │   └── skin_engine.py               # TUI skin/theming engine
│   │
│   ├── cron/                            # Cron Scheduler
│   │   ├── __init__.py
│   │   ├── scheduler.py                 # Cron job scheduler
│   │   └── jobs.py                      # Job management
│   │
│   ├── tasks/                           # Task Management
│   │   ├── __init__.py
│   │   ├── manager.py                   # Task manager
│   │   ├── executor.py                  # Task executor
│   │   └── queue.py                     # Task queue
│   │
│   ├── web/                             # Web Integration
│   │   ├── __init__.py
│   │   ├── fetch.py                     # Web content fetching
│   │   ├── search.py                    # Web search engine
│   │   └── links.py                     # Link resolution & preview
│   │
│   ├── realtime/                        # Real-time Processing
│   │   ├── __init__.py
│   │   ├── transcription.py             # Real-time transcription
│   │   └── voice.py                     # Voice processing
│   │
│   ├── i18n/                            # Internationalization
│   │   ├── __init__.py
│   │   ├── loader.py                    # Locale loader
│   │   └── locales/                     # Locale files
│   │       ├── __init__.py
│   │       └── en.json                  # English translations
│   │
│   ├── hooks/                           # Atlas Hook System
│   │   ├── __init__.py
│   │   └── system.py                    # System-level hooks
│   │
│   ├── polls/                           # Polls & Voting
│   │   ├── __init__.py
│   │   └── manager.py                   # Poll management
│   │
│   ├── pairing/                         # Device Pairing
│   │   ├── __init__.py
│   │   ├── discovery.py                 # Device discovery
│   │   └── manager.py                   # Pairing manager
│   │
│   ├── node_host/                       # Node Host (IoT/Edge)
│   │   ├── __init__.py
│   │   ├── device.py                    # Device management
│   │   ├── camera.py                    # Camera access
│   │   └── screen.py                    # Screen capture
│   │
│   └── link_understanding/              # Link Intelligence
│       ├── __init__.py
│       └── analyzer.py                  # URL/content analysis
│
├── cli/                                 # Claude Code Terminal Interface
│   ├── __init__.py
│   ├── app.py                           # CLI application (prompt_toolkit + rich)
│   └── renderer.py                      # Markdown + syntax highlighting renderer
│
├── gui/                                 # Cowork Desktop Interface
│   ├── __init__.py
│   ├── app.py                           # Desktop GUI application (tkinter)
│   ├── sidebar.py                       # File tree + task history sidebar
│   └── widgets.py                       # Custom tkinter widgets
│
├── plugins/                             # Plugin System
│   ├── __init__.py
│   └── loader.py                        # Hot-reloadable plugin loader
│
└── utils/                               # Shared Utilities
    ├── __init__.py
    ├── code_diff.py                     # Unified diff generation
    ├── database.py                      # Database helpers
    ├── git_manager.py                   # Git operations wrapper
    └── webhook.py                       # Webhook integration
```

---

## Module Descriptions

### `agent/` — Core AI Agent
The heart of the framework. Contains the agentic Think-Act-Observe-Iterate loop, 61+ tool implementations, 20 specialist teams, memory management, sandboxed execution, code analysis, security scanning, and desktop automation. The `self_improving/` sub-package provides 7 coordinated subsystems for self-analysis and auto-improvement.

### `crew/` — Multi-Agent Crew Orchestration
Enables coordinated multi-agent execution. Define crews of specialized agents, assign tasks with guardrails, control execution order (sequential or parallel), and track usage metrics. Includes a training handler for iterative crew improvement through repeated execution cycles.

### `flow/` — Workflow Orchestration
Decorator-based workflow engine for defining complex multi-step pipelines. Use `@start`, `@listen`, `@router`, `@and_`, and `@or_` decorators to wire steps together declaratively. Supports human-in-the-loop feedback gates, persistent state (SQLite backend), and automatic Mermaid/DOT diagram generation for visualization.

### `events/` — Event System
Central publish/subscribe event bus for cross-component communication. Provides typed events for Agent, Crew, Task, Tool, LLM, Memory, Flow, and System lifecycles. Components emit events; other components subscribe and react — enabling loose coupling between modules.

### `knowledge/` — Knowledge Sources
Unified interface for ingesting and querying knowledge from external sources. Supports PDF, CSV, Excel, JSON, plain text files, and raw strings as knowledge inputs. Designed to feed the agent's knowledge base for contextual retrieval.

### `llm/` — LLM Abstraction Layer
Multi-provider language model abstraction supporting Anthropic, OpenAI, Azure OpenAI, Google Gemini, Ollama, and any OpenAI-compatible API. Provides a consistent `LLMConfig`, `LLMResponse`, and `BaseLLM` interface regardless of the underlying provider.

### `hooks/` — Lifecycle Hooks
Before/after callback system for LLM calls and tool invocations. Use `@before_llm_call`, `@after_llm_call`, `@before_tool_call`, and `@after_tool_call` decorators to inject custom logic (logging, transformation, rate limiting, etc.) into the agent's execution pipeline.

### `evaluation/` — Agent Evaluation
Performance measurement and assessment framework for agents. Provides `AgentEvaluator`, `EvaluationResult`, and `EvaluationMetric` for quantifying agent quality across multiple dimensions.

### `telemetry/` — Telemetry & Metrics
Usage tracking and metrics collection for monitoring agent behavior, token consumption, tool usage patterns, and error rates over time.

### `security/` — Security Utilities
Agent fingerprinting for identity verification. Supports generating and verifying cryptographic fingerprints to ensure agent authenticity and integrity.

### `i18n/` — Internationalization
Localization support with locale loading and fallback chains. Currently includes English translations with infrastructure for 9 locales (en, es, zh, ja, ko, de, fr, pt, ru).

### `atlas/` — Atlas Agent Sub-system
A comprehensive agentic sub-system with 24 integrated subsystems, connected via the `AtlasBridge` facade. Includes context compression, smart routing, memory management, skills, plugin SDK, canvas UI, media pipeline, multi-platform gateway (24 messaging platforms), security, session management, ACP server, cron scheduling, and much more.

---

## Features

### Core Capabilities
- **Agentic AI Loop** — Think -> Act -> Observe -> Iterate (up to 10 rounds per message)
- **61+ Built-in Tools** — files, search, execution, web, git, databases, security, deployment, memory, desktop
- **20 Specialist Teams** — search, codegen, debug, review, test, refactor, docs, security, perf, devops, database, api, frontend, backend, data, architect, git, requirements, deploy, learn

### Multi-Agent Orchestration
- **Crew System** — Define crews of specialized agents with coordinated task execution
- **Guardrails** — Output validation and safety constraints on agent responses
- **Training** — Iterative crew improvement through training cycles with metrics tracking
- **Caching** — Response caching for performance optimization
- **Rate Limiting** — Per-agent RPM (requests per minute) control

### Workflow Automation
- **Flow Engine** — Decorator-based multi-step workflow definition
- **Conditional Routing** — `@router`, `@and_`, `@or_` for branching logic
- **Human Feedback** — Human-in-the-loop gates for critical decisions
- **Persistence** — SQLite-backed flow state storage
- **Visualization** — Auto-generate Mermaid/DOT flow diagrams

### Event-Driven Architecture
- **Central Event Bus** — Publish/subscribe pattern for loose coupling
- **8 Typed Event Categories** — Agent, Crew, Task, Tool, LLM, Memory, Flow, System
- **Priority Levels** — HIGH, NORMAL, LOW event priorities
- **Async Processing** — Non-blocking event propagation

### Atlas Agent Sub-system
- **Context Compression** — Compress conversations to fit 200K+ token contexts
- **Smart Routing** — Cost/latency-aware model selection
- **Memory Management** — Unified memory with 9 provider plugins (Mem0, ByteRover, Hindsight, etc.)
- **Skills System** — Discoverable, executable skills with 5 built-in skills
- **Plugin SDK** — Full plugin development framework with contracts, sandboxing, and registry
- **Canvas/A2UI** — Visual workspace with real-time widget push updates
- **Media Pipeline** — Audio, image, video processing and generation
- **Multi-Platform Gateway** — 24 messaging platform integrations (Slack, Discord, Telegram, Teams, etc.)
- **ACP Server** — Editor/IDE integration protocol (FastAPI)
- **Cron Scheduler** — Scheduled job management
- **Security** — Policy engine, audit logging, allowlisting, sandboxing, secrets management
- **Session Management** — Lifecycle management with persistence and transcripts
- **Internationalization** — 9 supported locales with fallback chains
- **Device Pairing** — Secure device discovery and pairing
- **Node Host** — IoT/Edge device management with camera and screen access
- **Link Understanding** — URL/content intelligence and analysis
- **Real-time Processing** — Live transcription and voice processing
- **Web Integration** — Content fetching, search, and link resolution

### Self-Improving System
- **Safety** — Guardrails, approval gates, backup/rollback, quarantine zone, protected files
- **Evaluator** — Deep static analysis, code quality scoring, bug detection
- **Patcher** — Automatic bug fix generation with verified application
- **Extender** — Generates new tools to fill capability gaps
- **Optimizer** — Performance profiling and bottleneck optimization
- **Learner** — User preference learning and behavior adaptation
- **Evolution** — Timeline tracking, improvement metrics, generational lineage

### Desktop Automation
- **Awareness** — Clipboard monitoring, window tracking, screenshots with OCR
- **Controller** — Smooth mouse movement, human-like typing, keyboard hotkeys
- **Voice** — Speech-to-text (Google/WSR), text-to-speech (pyttsx3), wake word
- **Permissions** — Granular levels (STANDARD/EXPERT/TRUSTED), audit logging

### Dual Interface
- **CLI Mode** — Terminal UI with prompt_toolkit + rich, streaming, autocomplete, slash commands, vim mode
- **GUI Mode** — Desktop app with tkinter, file tree, task checklist, settings dialog
- **Atlas TUI** — Full-featured terminal interface with skin engine, profiles, and skills hub

### Knowledge & Memory
- **Knowledge Base** — Graph-based persistent knowledge storage with semantic search
- **Memory** — SQLite-backed conversation memory with auto-summarization
- **Knowledge Import** — Import from Obsidian vaults, Markdown files, and structured data

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your API Key

```bash
# OpenRouter (recommended — supports 200+ models)
export OPENROUTER_API_KEY=sk-or-your-key-here

# Or Anthropic Direct
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Run

```bash
# Launch the desktop GUI
python main.py

# Launch the terminal CLI
python main.py --cli

# Enable Atlas Agent mode
python main.py --atlas-cli

# Run diagnostics
python main.py --doctor

# Show project statistics
python main.py --stats
```

### Using the Crew System

```python
from crew import Crew, CrewAgent, Task

# Define agents
researcher = CrewAgent(
    role="Researcher",
    goal="Find relevant information",
    backstory="You are an expert researcher.",
)

writer = CrewAgent(
    role="Writer",
    goal="Write compelling content",
    backstory="You are a skilled writer.",
)

# Define tasks
research_task = Task(
    description="Research the latest AI developments",
    agent=researcher,
)

write_task = Task(
    description="Write a summary based on the research",
    agent=writer,
    context=[research_task],  # Depends on research output
)

# Assemble and run crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
)
result = crew.run()
```

### Using the Flow System

```python
from flow import Flow, start, listen, router, and_

workflow = Flow("My Workflow")

@workflow.start
def begin(ctx):
    ctx["data"] = fetch_data()
    return "process"

@workflow.listen("process")
def process_data(ctx):
    ctx["result"] = analyze(ctx["data"])
    return "route"

@workflow.router("route")
def route_result(ctx):
    if ctx["result"]["score"] > 0.8:
        return "high_quality"
    return "low_quality"

@workflow.listen("high_quality")
def handle_high(ctx):
    deploy(ctx["result"])

@workflow.listen("low_quality")
def handle_low(ctx):
    request_review(ctx["result"])

result = workflow.run()
```

---

## Command Line Options

```
python main.py [OPTIONS]

Core:
  --cli                Launch CLI mode (Claude Code terminal)
  --vim                Enable vim keybindings (CLI mode)
  --model MODEL        AI model to use (default: anthropic/claude-sonnet-4-20250514)
  --theme THEME        Color theme: dark or light (default: dark)
  --max-tokens N       Max tokens per response (default: 8192)
  --max-iterations N   Max agent iterations per message (default: 10)
  --cwd PATH           Set working directory
  --provider PROVIDER  API provider: openrouter or anthropic (default: openrouter)

Agent:
  --agent AGENT        Start with a specific specialist agent
  --self-improve       Enable the self-improving system
  --knowledge-base     Enable the knowledge base system

Atlas:
  --atlas              Enable Atlas Agent mode
  --atlas-cli          Launch the Atlas TUI
  --gateway            Start the Atlas Gateway server
  --acp                Start the Atlas ACP server
  --security           Enable security policy enforcement
  --canvas             Enable Canvas/A2UI visual workspace
  --voice              Enable real-time voice mode
  --locale LOCALE      Set UI locale (e.g., en, es, zh)

System:
  --plugin-dir PATH    Load plugins from directory
  --sandbox-type TYPE  Sandbox type: none, docker, process, restricted
  --config-file PATH   Load config from file
  --doctor             Run full diagnostics
  --stats              Show project statistics
  --version            Show version info (v3.0.0)
```

---

## Project Statistics

| Metric | Value |
|--------|-------|
| **Python files** | 299 |
| **Lines of code** | 157,986 |
| **Top-level modules** | 12 (agent, atlas, cli, crew, evaluation, events, flow, gui, hooks, i18n, knowledge, llm, plugins, security, telemetry, utils) |
| **Atlas subsystems** | 24 |
| **Atlas tools** | 30 |
| **Agent tools** | 61+ |
| **Specialist teams** | 20 |
| **Built-in skills** | 5 |
| **Platform integrations** | 24 (Slack, Discord, Telegram, WhatsApp, Teams, Signal, IRC, Matrix, Twitch, etc.) |
| **Memory plugins** | 9 |
| **Event types** | 8 categories |
| **Supported locales** | 9 |
| **Self-improving subsystems** | 7 |

---

## License

MIT
