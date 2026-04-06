"""
Agent core — the agentic loop that drives Claude Clone.

Implements:
- Full streaming agentic loop with Anthropic SDK
- Typed events: ThinkingEvent, TextEvent, ToolCallEvent, ToolResultEvent, ErrorEvent, DoneEvent
- Multi-turn conversation with message history
- Tool use → tool result → continue loop
- Max iteration guard
- Smart context injection (cwd, OS, git status, open files, project type)
"""

import asyncio
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from agent.tools import TOOLS_REGISTRY, generate_tool_schemas, set_self_improving_orchestrator, load_atlas_tools, get_atlas_tool_schemas
from agent.sandbox import SandboxExecutor
from agent.memory import ConversationMemory, get_memory
from agent.analyzer import ProjectAnalyzer
from agent.security import SecurityScanner
from agent.self_improving import SelfImprovingOrchestrator
from agent.knowledge_base import KnowledgeBaseOrchestrator, get_knowledge_base, set_knowledge_base

# ── Atlas Plugin/Hook adapter (replaces legacy plugins.loader) ──
_AtlasPluginLoader = None
_AtlasHookSystem = None
_AtlasPluginRegistry = None


def _init_atlas_plugin_system():
    """Lazily import Atlas plugin SDK and hook system."""
    global _AtlasPluginLoader, _AtlasPluginHookSystem, _AtlasPluginRegistry
    try:
        from atlas.plugin_sdk import PluginLoader, PluginRegistry as _PR
        from atlas.hooks.system import HookSystem
        _AtlasPluginLoader = PluginLoader
        _AtlasPluginHookSystem = HookSystem
        _AtlasPluginRegistry = _PR
        return True
    except ImportError:
        return False


class PluginManager:
    """
    Compatibility adapter that delegates to Atlas PluginLoader + HookSystem.

    This replaces the legacy root-level plugins.loader.PluginManager.
    All hook/tool logic now flows through atlas.plugin_sdk and atlas.hooks.
    """

    def __init__(self):
        self._loader = None
        self._hook_system = None
        self._registry = None
        _init_atlas_plugin_system()
        if _AtlasPluginLoader is not None:
            self._registry = _AtlasPluginRegistry()
            self._loader = _AtlasPluginLoader(registry=self._registry)
        if _AtlasPluginHookSystem is not None:
            self._hook_system = _AtlasPluginHookSystem()

    async def load_all(self):
        """Load all discovered plugins (sync in atlas, wrapped for compat)."""
        if self._loader:
            self._loader.load_all()

    def get_tools(self):
        """Return plugin tools as {name: callable} dict for Agent.tools."""
        if not self._loader:
            return {}
        tools = {}
        all_tools = self._loader.get_all_tools()
        for plugin_name, tool_defs in all_tools.items():
            for td in tool_defs:
                if hasattr(td, 'handler') and callable(td.handler):
                    tools[td.name] = td.handler
                elif hasattr(td, 'function') and callable(td.function):
                    tools[td.name] = td.function
        return tools

    async def execute_hook(self, hook_name: str, data: dict):
        """Execute a hook by string name (e.g. 'PRE_EXECUTION')."""
        if not self._hook_system:
            return
        try:
            from atlas.hooks.system import HookPoint, HookContext
            # Map string names to HookPoint enum values
            hook_map = {
                "PRE_EXECUTION": HookPoint.PRE_EXECUTION,
                "POST_EXECUTION": HookPoint.POST_EXECUTION,
                "PRE_TOOL_CALL": HookPoint.PRE_TOOL_CALL,
                "POST_TOOL_CALL": HookPoint.POST_TOOL_CALL,
                "ON_ERROR": HookPoint.ON_ERROR,
                "ON_MESSAGE": HookPoint.ON_MESSAGE,
                "ON_RESPONSE": HookPoint.ON_RESPONSE,
                "PRE_SEND": HookPoint.PRE_SEND,
                "POST_SEND": HookPoint.POST_SEND,
                "ON_CONNECT": HookPoint.ON_CONNECT,
                "ON_DISCONNECT": HookPoint.ON_DISCONNECT,
                "SESSION_START": HookPoint.SESSION_START,
                "SESSION_END": HookPoint.SESSION_END,
                "CONFIG_CHANGE": HookPoint.CONFIG_CHANGE,
                "PLUGIN_LOAD": HookPoint.PLUGIN_LOAD,
                "PLUGIN_UNLOAD": HookPoint.PLUGIN_UNLOAD,
            }
            hp = hook_map.get(hook_name)
            if hp is not None:
                ctx = HookContext(hook_point=hp, data=data)
                await self._hook_system.execute(hp, ctx)
        except ImportError:
            pass
        except Exception:
            pass

    def list_active(self):
        """Return list of active plugin names."""
        if not self._registry:
            return []
        return [p.name for p in self._registry.list_active()]


# ──────────────────────────────────────────────
# Event types
# ──────────────────────────────────────────────

class AgentEvent:
    """Base event emitted by the agent during execution."""
    event_type: str = "base"

    def __init__(self, data: Any = None):
        self.data = data
        self.timestamp = datetime.now().isoformat()

    def __repr__(self):
        return f"<{self.__class__.__name__} data={self.data!r:.100}>"


class ThinkingEvent(AgentEvent):
    """The model is thinking (extended thinking)."""
    event_type = "thinking"


class TextEvent(AgentEvent):
    """A chunk of text output from the model."""
    event_type = "text"


class ToolCallEvent(AgentEvent):
    """The model wants to call a tool."""
    event_type = "tool_call"

    def __init__(self, tool_name: str, tool_input: dict, tool_id: str):
        super().__init__(data={"name": tool_name, "input": tool_input, "id": tool_id})
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.tool_id = tool_id


class ToolResultEvent(AgentEvent):
    """Result from a tool execution."""
    event_type = "tool_result"

    def __init__(self, tool_name: str, result: str, tool_id: str, is_error: bool = False):
        super().__init__(data={"name": tool_name, "result": result, "id": tool_id, "is_error": is_error})
        self.tool_name = tool_name
        self.result = result
        self.tool_id = tool_id
        self.is_error = is_error


class ErrorEvent(AgentEvent):
    """An error occurred during agent execution."""
    event_type = "error"


class DoneEvent(AgentEvent):
    """The agent has finished processing."""
    event_type = "done"

    def __init__(self, usage: dict = None, total_cost: float = 0.0):
        super().__init__(data=usage)
        self.usage = usage or {}
        self.total_cost = total_cost


class UsageEvent(AgentEvent):
    """Token usage update."""
    event_type = "usage"

    def __init__(self, input_tokens: int, output_tokens: int):
        super().__init__(data={"input_tokens": input_tokens, "output_tokens": output_tokens})
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


# ──────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────

class Agent:
    """
    The main agentic loop that orchestrates Claude API calls with tool use.

    Usage:
        agent = Agent(api_key="sk-ant-...", tools=TOOLS_REGISTRY)
        async for event in agent.run("Read main.py and explain it"):
            print(event)
    """

    DEFAULT_SYSTEM_PROMPT = """You are an expert agentic AI assistant running inside a local Python environment.
You have full access to the user's file system and terminal via tools.

## BEHAVIOR
- Think step by step before acting.
- Prefer doing over asking. Attempt the task, then report.
- Chain tool calls together — do not stop and ask after every step.
- After a tool call, review output and decide the next action.
- If a command fails, diagnose and retry with corrections.
- Only ask the user when you are truly blocked.

## CONTEXT
{context}

## RULES
- Never delete files without confirmation.
- Never expose secrets or API keys.
- Never run destructive commands without approval.
- Always read a file before editing it.
- After writing code, run it to verify it works.
- When editing files, use edit_file with exact text matching.
- Provide clear, concise responses with actionable information.
"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "anthropic/claude-sonnet-4-20250514",
        system_prompt: str = None,
        tools: Dict[str, Callable] = None,
        max_tokens: int = 8192,
        max_iterations: int = 10,
        temperature: float = 1.0,
        cost_callback: Callable = None,
        base_url: str = None,
        sandbox: bool = True,
        memory: bool = True,
        analyzer: bool = True,
        plugins: bool = True,
        self_improving: bool = False,
        knowledge_base: bool = False,
        atlas_mode: bool = False,
        project_root: str = None,
        atlas_config: dict = None,
    ):
        # API key: OpenRouter first, then Anthropic
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.tools = tools or {}
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.cost_callback = cost_callback
        self.base_url = base_url  # None = use SDK default (Anthropic direct)

        # Feature flags
        self.sandbox_enabled = sandbox
        self.memory_enabled = memory
        self.analyzer_enabled = analyzer
        self.plugins_enabled = plugins
        self.self_improving_enabled = self_improving

        self.messages: List[Dict] = []
        self.context_files: List[str] = []
        self._client = None
        self._cancelled = False
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Sandbox executor (eager init)
        self.sandbox_executor = SandboxExecutor() if sandbox else None

        # Conversation memory (eager init)
        self.conversation_memory = get_memory() if memory else None

        # Project analyzer (lazy init)
        self.project_analyzer: Optional[ProjectAnalyzer] = None
        if analyzer:
            try:
                self.project_analyzer = ProjectAnalyzer()
            except Exception:
                pass  # Gracefully degrade if analyzer unavailable

        # Security scanner (lazy init)
        self.security_scanner: Optional[SecurityScanner] = None

        # Plugin manager (lazy init)
        self.plugin_manager: Optional[PluginManager] = None

        # Self-improving orchestrator (lazy init)
        self.self_improving_orchestrator: Optional[SelfImprovingOrchestrator] = None
        if self_improving:
            try:
                root = project_root or os.getcwd()
                self.self_improving_orchestrator = SelfImprovingOrchestrator(
                    agent=self,
                    project_root=root,
                )
            except Exception:
                pass  # Non-critical: degrade gracefully

        # Knowledge base (lazy init)
        self.knowledge_base_enabled = knowledge_base
        self._kb_orchestrator: Optional[KnowledgeBaseOrchestrator] = None
        if knowledge_base:
            try:
                self._kb_orchestrator = KnowledgeBaseOrchestrator()
                set_knowledge_base(self._kb_orchestrator)
            except Exception:
                pass  # Non-critical: degrade gracefully

        # ── Atlas Agent Integration ──
        self.atlas_mode = atlas_mode
        self.atlas_config = atlas_config or {}
        self._atlas_compressor = None
        self._atlas_prompt_builder = None
        self._atlas_memory_manager = None
        self._atlas_smart_router = None
        self._atlas_credential_pool = None
        self._atlas_insights = None
        self._atlas_trajectory = None

        if atlas_mode:
            try:
                # Load Atlas context compressor
                if self.atlas_config.get("context_compression", {}).get("enabled", True):
                    from atlas.core.context_compressor import ContextCompressor
                    self._atlas_compressor = ContextCompressor(
                        strategy=self.atlas_config.get("context_compression", {}).get("strategy", "summarize"),
                    )

                # Load Atlas prompt builder
                if self.atlas_config.get("prompt_builder", {}).get("enabled", True):
                    from atlas.core.prompt_builder import PromptBuilder
                    self._atlas_prompt_builder = PromptBuilder()

                # Load Atlas memory manager
                from atlas.core.memory_manager import MemoryManager
                from atlas.core.builtin_memory import BuiltinMemoryProvider
                builtin_provider = BuiltinMemoryProvider()
                self._atlas_memory_manager = MemoryManager(builtin_provider)

                # Load Atlas smart router
                if self.atlas_config.get("smart_routing", {}).get("enabled", True):
                    from atlas.core.smart_routing import SmartRouter
                    self._atlas_smart_router = SmartRouter()

                # Load Atlas credential pool (optional)
                if self.atlas_config.get("credential_pool", {}).get("enabled", False):
                    from atlas.core.credential_pool import CredentialPool
                    self._atlas_credential_pool = CredentialPool(
                        strategy=self.atlas_config.get("credential_pool", {}).get("strategy", "round_robin"),
                    )

                # Load Atlas insights
                if self.atlas_config.get("insights", {}).get("enabled", True):
                    from atlas.core.insights import InsightsManager
                    self._atlas_insights = InsightsManager()

                # Load Atlas trajectory recorder (optional)
                if self.atlas_config.get("trajectory", {}).get("enabled", False):
                    from atlas.core.trajectory import TrajectoryRecorder
                    self._atlas_trajectory = TrajectoryRecorder(
                        storage_path=self.atlas_config.get("trajectory", {}).get("storage_path"),
                    )

                # Merge Atlas tools into the existing tool registry
                atlas_tools = load_atlas_tools()
                if atlas_tools:
                    self.tools.update(atlas_tools)
                    self.tool_schemas = generate_tool_schemas(self.tools)

                # Initialize Atlas memory plugins if configured
                if self.atlas_config.get("memory_plugins", {}).get("enabled", False):
                    try:
                        from atlas.plugins.memory.registry import MemoryPluginRegistry
                        mem_registry = MemoryPluginRegistry()
                        mem_plugin_configs = self.atlas_config.get("memory_plugins", {}).get("plugins", [])
                        for plugin_conf in mem_plugin_configs:
                            try:
                                mem_registry.register(plugin_conf)
                            except Exception:
                                pass
                        if self._atlas_memory_manager and hasattr(mem_registry, 'get_providers'):
                            for provider in mem_registry.get_providers():
                                try:
                                    self._atlas_memory_manager.register_provider(provider)
                                except Exception:
                                    pass
                    except ImportError:
                        pass

            except ImportError as e:
                import logging
                logging.getLogger(__name__).warning(f"Atlas mode requested but import failed: {e}")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Atlas initialization failed: {e}")

        # Cached project analysis result
        self._project_analysis: Optional[dict] = None

        # Tool schemas
        self.tool_schemas = generate_tool_schemas(self.tools) if self.tools else []

        # Register self-improving orchestrator with tools layer
        if self.self_improving_orchestrator:
            set_self_improving_orchestrator(self.self_improving_orchestrator)

    def _get_client(self):
        """Lazy-initialize the Anthropic client (supports OpenRouter bypass)."""
        if self._client is None:
            try:
                import anthropic
                # Build kwargs for the client
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    # OpenRouter (or any compatible provider) — point to custom base_url
                    client_kwargs["base_url"] = self.base_url
                    # OpenRouter requires HTTP headers for routing
                    client_kwargs["default_headers"] = {
                        "HTTP-Referer": "https://github.com/claude-clone",
                        "X-Title": "Claude Clone",
                    }
                self._client = anthropic.AsyncAnthropic(**client_kwargs)
            except ImportError:
                raise ImportError("anthropic package is required. Install it with: pip install anthropic")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize API client: {e}")
        return self._client

    def reset(self):
        """Clear conversation history."""
        self.messages = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def add_context(self, path: str) -> str:
        """Add a file's content to the agent's context."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"File not found: {p}"
        if not p.is_file():
            return f"Not a file: {p}"
        if path not in self.context_files:
            self.context_files.append(str(p))
            return f"Added {p} to context"
        return f"{p} already in context"

    def remove_context(self, path: str) -> str:
        """Remove a file from context."""
        p = Path(path).expanduser().resolve()
        if str(p) in self.context_files:
            self.context_files.remove(str(p))
            return f"Removed {p} from context"
        return f"{p} not in context"

    def cancel(self):
        """Cancel the current generation."""
        self._cancelled = True

    def _build_context_string(self) -> str:
        """Build the context string to inject into the system prompt."""
        parts = []

        # Current working directory
        try:
            cwd = os.getcwd()
        except Exception:
            cwd = "unknown"
        parts.append(f"- CWD: {cwd}")

        # OS info
        parts.append(f"- OS: {platform.system()} {platform.release()} ({platform.machine()})")
        parts.append(f"- Python: {sys.version.split()[0]}")

        # Datetime
        parts.append(f"- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Git status
        git_status = self._get_git_info()
        if git_status:
            parts.append(f"- Git: {git_status}")

        # Open files
        if self.context_files:
            files_list = ", ".join(os.path.basename(f) for f in self.context_files)
            parts.append(f"- Open files in context: {files_list}")

        # Project type detection
        project_type = self._detect_project_type()
        if project_type:
            parts.append(f"- Project type: {project_type}")

        # Available tools
        if self.tool_schemas:
            tool_names = [s["name"] for s in self.tool_schemas]
            parts.append(f"- Available tools: {', '.join(tool_names)}")

        # Recent memory context (placeholder — populated async in run)
        parts.append("- Memory: available" if self.memory_enabled and self.conversation_memory else "- Memory: disabled")

        # Project analysis summary
        if self._project_analysis and self._project_analysis.get("status") == "ok":
            summary = self._project_analysis.get("summary", "")
            if summary:
                parts.append(f"- Project analysis: {summary[:300]}")

        # Active plugins list
        if self.plugin_manager and hasattr(self.plugin_manager, "list_active"):
            active = self.plugin_manager.list_active()
            if active:
                parts.append(f"- Active plugins: {', '.join(active)}")

        return "\n".join(parts)

    def _get_git_info(self) -> str:
        """Get git branch and status info."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return ""

            branch = result.stdout.strip()

            result2 = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            if result2.returncode == 0:
                dirty = len(result2.stdout.strip().split("\n"))
                if dirty > 0:
                    return f"branch={branch}, {dirty} dirty file(s)"
                return f"branch={branch}, clean"

            return f"branch={branch}"
        except Exception:
            return ""

    def _detect_project_type(self) -> str:
        """Detect project type based on files present."""
        cwd = Path.cwd()
        indicators = {
            "Python": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
            "Node.js": ["package.json", "yarn.lock", "pnpm-lock.yaml"],
            "Rust": ["Cargo.toml"],
            "Go": ["go.mod", "go.sum"],
            "Java/Kotlin": ["pom.xml", "build.gradle", "build.gradle.kts"],
            "Ruby": ["Gemfile", "Rakefile"],
            "PHP": ["composer.json"],
            "C/C++": ["CMakeLists.txt", "Makefile"],
        }

        detected = []
        for project, files in indicators.items():
            for f in files:
                if (cwd / f).exists():
                    detected.append(project)
                    break

        return ", ".join(detected) if detected else ""

    # ──────────────────────────────────────────────
    # New integration methods
    # ──────────────────────────────────────────────

    async def initialize_plugins(self) -> None:
        """Initialize the plugin manager, load all plugins, and merge plugin tools."""
        if not self.plugins_enabled:
            return
        try:
            self.plugin_manager = PluginManager()
            await self.plugin_manager.load_all()
            plugin_tools = self.plugin_manager.get_tools()
            if plugin_tools:
                self.tools.update(plugin_tools)
                self.tool_schemas = generate_tool_schemas(self.tools)
        except Exception as e:
            # Plugins are non-critical; log and continue
            pass  # Silently degrade — plugin errors are non-critical

    async def get_memory_context(self, query: str, max_tokens: int = 2000) -> str:
        """Retrieve relevant memories for context injection."""
        if not self.memory_enabled or not self.conversation_memory:
            return ""
        try:
            memories = await self.conversation_memory.search(query, limit=5)
            if not memories:
                return ""
            lines = [f"- [memory] {m['content'][:200]}" for m in memories[:5]]
            return "\n".join(lines)
        except Exception:
            return ""

    async def analyze_current_project(self) -> dict:
        """Analyze the current working directory for project structure and metadata."""
        if not self.analyzer_enabled or not self.project_analyzer:
            return {"status": "disabled"}
        try:
            self._project_analysis = await self.project_analyzer.analyze(os.getcwd())
            return self._project_analysis
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def security_scan_current(self) -> dict:
        """Run a security scan on the current directory."""
        if self.security_scanner is None:
            try:
                self.security_scanner = SecurityScanner()
            except Exception as e:
                return {"status": "error", "error": f"SecurityScanner init failed: {e}"}
        try:
            return await self.security_scanner.scan(os.getcwd())
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def enable_atlas_cron(self) -> None:
        """Start the Atlas CronScheduler in a background thread.

        Loads scheduled jobs from the Atlas configuration and runs them
        in a daemon thread. Non-critical: degrades gracefully if Atlas is
        not installed or the scheduler fails to start.
        """
        try:
            from atlas.cron.scheduler import CronScheduler
            from atlas.cron.jobs import JobManager
            job_manager = JobManager()
            scheduler = CronScheduler(job_manager=job_manager)
            import threading
            thread = threading.Thread(target=scheduler.run, daemon=True, name="atlas-cron")
            thread.start()
            self._atlas_cron_scheduler = scheduler
            self._atlas_cron_thread = thread
        except ImportError:
            pass  # Atlas cron not available
        except Exception:
            pass  # Non-critical

    async def enable_atlas_skills(self) -> None:
        """Initialize the Atlas SkillManager and load built-in skills.

        Merges any tool functions exposed by skills into the agent's tool
        registry. Non-critical: degrades gracefully.
        """
        try:
            from atlas.skills.manager import SkillManager
            skill_mgr = SkillManager()
            # Load built-in skills
            if hasattr(skill_mgr, 'load_builtins'):
                skill_mgr.load_builtins()
            # Merge skill tools into agent tools
            skill_tools = skill_mgr.get_tools() if hasattr(skill_mgr, 'get_tools') else {}
            if skill_tools:
                self.tools.update(skill_tools)
                self.tool_schemas = generate_tool_schemas(self.tools)
            self._atlas_skill_manager = skill_mgr
        except ImportError:
            pass  # Atlas skills not available
        except Exception:
            pass  # Non-critical

    async def execute_in_sandbox(self, code: str, language: str = "python") -> str:
        """Execute code in an isolated sandbox environment."""
        if not self.sandbox_enabled or not self.sandbox_executor:
            return "Error: Sandbox is not enabled."
        try:
            result = await self.sandbox_executor.execute(code, language=language)
            return result.get("stdout", "") if isinstance(result, dict) else str(result)
        except Exception as e:
            return f"Sandbox execution error: {e}"

    async def _build_context_files_content(self) -> str:
        """Read context files and return their content."""
        parts = []
        for path_str in self.context_files:
            try:
                p = Path(path_str)
                if p.exists() and p.is_file():
                    content = p.read_text(encoding="utf-8", errors="replace")
                    # Truncate very large files
                    max_chars = 50000
                    if len(content) > max_chars:
                        content = content[:max_chars] + f"\n\n[... truncated, {len(content)} total chars]"
                    rel = p.relative_to(Path.cwd()) if p.is_relative_to(Path.cwd()) else p.name
                    parts.append(f"--- File: {rel} ---\n{content}\n--- End of {rel} ---\n")
            except Exception as e:
                parts.append(f"--- File: {path_str} ---\nError reading: {e}\n")
        return "\n".join(parts)

    async def run(self, user_message: str) -> AsyncIterator[AgentEvent]:
        """
        Run the agent with a user message. Yields AgentEvent objects.

        This is the main entry point. It:
        1. Adds the user message to conversation history
        2. Injects context into system prompt
        3. Runs the agentic loop
        """
        self._cancelled = False

        # Initialize plugins if enabled (PRE_EXECUTION hook)
        if self.plugins_enabled and self.plugin_manager is None:
            await self.initialize_plugins()
        if self.plugin_manager:
            try:
                await self.plugin_manager.execute_hook("PRE_EXECUTION", {"message": user_message})
            except Exception:
                pass

        # Build system message with context
        context_str = self._build_context_string()

        # Inject recent memory context
        if self.memory_enabled and self.conversation_memory:
            memory_ctx = await self.get_memory_context(user_message, max_tokens=2000)
            if memory_ctx:
                context_str += f"\n\n## RELEVANT MEMORIES\n{memory_ctx}"

        # Inject knowledge base context
        if self.knowledge_base_enabled and self._kb_orchestrator:
            try:
                if not self._kb_orchestrator.initialized:
                    await self._kb_orchestrator.initialize()
                kb_ctx = await self._kb_orchestrator.get_context_for_prompt(user_message, max_tokens=2000)
                if kb_ctx:
                    context_str += f"\n\n## KNOWLEDGE BASE\n{kb_ctx}"
            except Exception:
                pass  # Non-critical

        # Add context file contents if any
        if self.context_files:
            ctx_content = await self._build_context_files_content()
            context_str += f"\n\n## FILES IN CONTEXT\n{ctx_content}"

        system_msg = self.system_prompt.format(context=context_str)

        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": user_message,
        })

        # ── Atlas: Context compression if history is too long ──
        if self.atlas_mode and self._atlas_compressor and len(self.messages) > 10:
            try:
                self.messages = await self._atlas_compressor.compress(self.messages, self.model)
            except Exception:
                pass  # Non-critical: proceed with uncompressed history

        # ── Atlas: Record insights after context compression ──
        if self.atlas_mode and self._atlas_insights:
            try:
                self._atlas_insights.record_event({
                    "type": "run_start",
                    "message_length": len(user_message),
                    "history_length": len(self.messages),
                    "model": self.model,
                })
            except Exception:
                pass  # Non-critical

        # ── Atlas: Record trajectory if enabled ──
        if self.atlas_mode and self._atlas_trajectory:
            try:
                self._atlas_trajectory.record_user_message(user_message)
            except Exception:
                pass

        # Run the agentic loop
        final_usage = {"input_tokens": 0, "output_tokens": 0}
        async for event in self._agentic_loop(system_msg):
            if isinstance(event, DoneEvent):
                final_usage = event.usage or {}
            yield event

        # POST_EXECUTION hook
        if self.plugin_manager:
            try:
                await self.plugin_manager.execute_hook("POST_EXECUTION", {
                    "message": user_message,
                    "usage": final_usage,
                })
            except Exception:
                pass

        # Auto-start self-improving orchestrator if enabled
        if self.self_improving_enabled and self.self_improving_orchestrator and not self.self_improving_orchestrator._initialized:
            try:
                await self.self_improving_orchestrator.initialize()
            except Exception:
                pass  # Non-critical

        # Save conversation to memory after completion
        if self.memory_enabled and self.conversation_memory:
            try:
                await self.conversation_memory.save_turn(
                    user_message=user_message,
                    assistant_messages=[m for m in self.messages if m["role"] == "assistant"],
                    metadata={"model": self.model, "usage": final_usage},
                )
            except Exception:
                pass  # Memory save is non-critical

    async def _agentic_loop(self, system_message: str) -> AsyncIterator[AgentEvent]:
        """
        The core agentic loop:
        1. Send messages to Claude with tools
        2. Stream the response
        3. If tool calls, execute them and continue
        4. Repeat until done or max iterations
        """
        client = self._get_client()

        for iteration in range(self.max_iterations):
            if self._cancelled:
                yield ErrorEvent(data="Generation cancelled by user")
                return

            # Build API params
            api_params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system_message,
                "messages": self.messages,
            }

            if self.tool_schemas:
                api_params["tools"] = self.tool_schemas

            try:
                # Stream the response
                current_text = ""
                tool_calls = []
                current_tool_id = None
                current_tool_name = None
                current_tool_input_json = ""

                async with client.messages.stream(**api_params) as stream:
                    async for event in stream:
                        if self._cancelled:
                            yield ErrorEvent(data="Generation cancelled by user")
                            return

                        if event.type == "content_block_start":
                            if hasattr(event, "content_block") and event.content_block:
                                if event.content_block.type == "thinking":
                                    yield ThinkingEvent(data=event.content_block.thinking if hasattr(event.content_block, "thinking") else "")
                                elif event.content_block.type == "tool_use":
                                    current_tool_id = event.content_block.id
                                    current_tool_name = event.content_block.name
                                    current_tool_input_json = ""

                        elif event.type == "content_block_delta":
                            if hasattr(event, "delta") and event.delta:
                                if event.delta.type == "text_delta":
                                    chunk = event.delta.text
                                    current_text += chunk
                                    yield TextEvent(data=chunk)
                                elif event.delta.type == "thinking_delta":
                                    yield ThinkingEvent(data=event.delta.thinking if hasattr(event.delta, "thinking") else "")
                                elif event.delta.type == "input_json_delta":
                                    if event.delta.partial_json:
                                        current_tool_input_json += event.delta.partial_json

                        elif event.type == "content_block_stop":
                            if current_tool_name and current_tool_id:
                                try:
                                    tool_input = json.loads(current_tool_input_json) if current_tool_input_json else {}
                                except json.JSONDecodeError:
                                    tool_input = {}
                                    yield ErrorEvent(data=f"Failed to parse tool input JSON for {current_tool_name}")

                                tool_calls.append({
                                    "id": current_tool_id,
                                    "name": current_tool_name,
                                    "input": tool_input,
                                })
                                yield ToolCallEvent(
                                    tool_name=current_tool_name,
                                    tool_input=tool_input,
                                    tool_id=current_tool_id,
                                )

                            current_tool_id = None
                            current_tool_name = None
                            current_tool_input_json = ""

                        elif event.type == "message_start":
                            if hasattr(event, "message") and event.message and hasattr(event.message, "usage"):
                                yield UsageEvent(
                                    input_tokens=event.message.usage.input_tokens,
                                    output_tokens=0,
                                )

                        elif event.type == "message_delta":
                            if hasattr(event, "usage"):
                                yield UsageEvent(
                                    input_tokens=0,
                                    output_tokens=event.usage.output_tokens,
                                )

                        elif event.type == "message_stop":
                            pass

                # Get the final message from stream
                final_message = await stream.get_final_message()

                # Update token counts
                if hasattr(final_message, "usage") and final_message.usage:
                    self._total_input_tokens += final_message.usage.input_tokens
                    self._total_output_tokens += final_message.usage.output_tokens

                # Add assistant message to history
                self.messages.append({
                    "role": "assistant",
                    "content": final_message.content,
                })

                # If no tool calls, we're done
                if not tool_calls:
                    yield DoneEvent(
                        usage={
                            "input_tokens": self._total_input_tokens,
                            "output_tokens": self._total_output_tokens,
                        }
                    )
                    return

                # Execute tool calls and build tool results
                tool_results = []
                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_input = tc["input"]
                    tool_id = tc["id"]

                    if tool_name in self.tools:
                        try:
                            # PRE_TOOL_CALL hook
                            if self.plugin_manager:
                                try:
                                    await self.plugin_manager.execute_hook("PRE_TOOL_CALL", {
                                        "tool_name": tool_name,
                                        "tool_input": tool_input,
                                        "tool_id": tool_id,
                                    })
                                except Exception:
                                    pass

                            # Execute the tool
                            tool_func = self.tools[tool_name]
                            # Pass dict kwargs directly
                            result = await tool_func(**tool_input)
                            if isinstance(result, dict):
                                result_str = json.dumps(result, indent=2, default=str)
                            else:
                                result_str = str(result)

                            # Truncate very long results
                            max_result_chars = 30000
                            if len(result_str) > max_result_chars:
                                result_str = result_str[:max_result_chars] + f"\n\n[... truncated, {len(result_str)} total chars]"

                            # POST_TOOL_CALL hook
                            if self.plugin_manager:
                                try:
                                    await self.plugin_manager.execute_hook("POST_TOOL_CALL", {
                                        "tool_name": tool_name,
                                        "tool_input": tool_input,
                                        "tool_id": tool_id,
                                        "result": result_str[:1000],
                                    })
                                except Exception:
                                    pass

                            yield ToolResultEvent(
                                tool_name=tool_name,
                                result=result_str,
                                tool_id=tool_id,
                                is_error=False,
                            )
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result_str,
                            })

                        except TypeError as e:
                            error_msg = f"Error calling {tool_name}: {e}\nCheck that the tool arguments are correct."
                            yield ToolResultEvent(tool_name=tool_name, result=error_msg, tool_id=tool_id, is_error=True)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": error_msg,
                                "is_error": True,
                            })
                        except Exception as e:
                            error_msg = f"Error calling {tool_name}: {e}"

                            # ON_ERROR hook
                            if self.plugin_manager:
                                try:
                                    await self.plugin_manager.execute_hook("ON_ERROR", {
                                        "tool_name": tool_name,
                                        "tool_input": tool_input,
                                        "tool_id": tool_id,
                                        "error": error_msg,
                                    })
                                except Exception:
                                    pass

                            yield ToolResultEvent(tool_name=tool_name, result=error_msg, tool_id=tool_id, is_error=True)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": error_msg,
                                "is_error": True,
                            })
                    else:
                        error_msg = f"Unknown tool: {tool_name}. Available tools: {', '.join(self.tools.keys())}"
                        yield ToolResultEvent(tool_name=tool_name, result=error_msg, tool_id=tool_id, is_error=True)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": error_msg,
                            "is_error": True,
                        })

                # Add tool results to conversation
                self.messages.append({
                    "role": "user",
                    "content": tool_results,
                })

                # Continue the loop to process tool results

            except Exception as e:
                error_str = str(e)

                # ON_ERROR hook for agentic loop errors
                if self.plugin_manager:
                    try:
                        await self.plugin_manager.execute_hook("ON_ERROR", {
                            "error": error_str,
                            "stage": "agentic_loop",
                        })
                    except Exception:
                        pass

                if "api_key" in error_str.lower() or "authentication" in error_str.lower():
                    yield ErrorEvent(data=f"Authentication error: Check your API key. {e}")
                elif "rate" in error_str.lower():
                    yield ErrorEvent(data=f"Rate limited: Please wait and try again. {e}")
                elif "timeout" in error_str.lower():
                    yield ErrorEvent(data=f"Request timeout: The API request took too long. {e}")
                else:
                    yield ErrorEvent(data=f"API error: {e}")
                yield DoneEvent(
                    usage={
                        "input_tokens": self._total_input_tokens,
                        "output_tokens": self._total_output_tokens,
                    }
                )
                return

        # Max iterations reached
        yield ErrorEvent(
            data=f"Reached maximum of {self.max_iterations} iterations. "
                 f"Type your message to continue or /clear to start over."
        )
        yield DoneEvent(
            usage={
                "input_tokens": self._total_input_tokens,
                "output_tokens": self._total_output_tokens,
            }
        )

    def get_conversation(self) -> List[Dict]:
        """Get the full conversation history."""
        return self.messages

    def get_token_counts(self) -> Dict[str, int]:
        """Get cumulative token usage."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }

    def estimate_cost(self) -> float:
        """Estimate total cost based on model pricing."""
        # Strip provider prefix (e.g. "anthropic/claude-sonnet-4" → "claude-sonnet-4")
        model_key = self.model
        if "/" in model_key:
            model_key = model_key.split("/")[-1]

        pricing = {
            "claude-opus-4-20250514": (15.0, 75.0),
            "claude-sonnet-4-20250514": (3.0, 15.0),
            "claude-3-5-sonnet-20241022": (3.0, 15.0),
            "claude-3-5-haiku-20241022": (0.8, 4.0),
            "claude-3-opus-20240229": (15.0, 75.0),
            "claude-3-sonnet-20240229": (3.0, 15.0),
            "claude-3-haiku-20240307": (0.25, 1.25),
        }
        input_cost, output_cost = pricing.get(model_key, (3.0, 15.0))
        return (
            (self._total_input_tokens / 1_000_000) * input_cost
            + (self._total_output_tokens / 1_000_000) * output_cost
        )

    def export_conversation(self) -> str:
        """Export conversation to markdown format."""
        lines = [f"# Claude Clone Conversation", ""]
        lines.append(f"**Model:** {self.model}")
        lines.append(f"**Tokens:** {self.get_token_counts()}")
        lines.append(f"**Estimated Cost:** ${self.estimate_cost():.4f}")
        lines.append(f"**Exported:** {datetime.now().isoformat()}")
        lines.append("")

        for msg in self.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, list):
                # Handle tool results
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            lines.append(f"## {role.title()}\n")
                            lines.append(block.get("text", ""))
                            lines.append("")
                        elif block.get("type") == "tool_use":
                            lines.append(f"### Tool Call: `{block.get('name', '')}`\n")
                            lines.append(f"```json\n{json.dumps(block.get('input', {}), indent=2)}\n```")
                            lines.append("")
                        elif block.get("type") == "tool_result":
                            lines.append(f"### Tool Result\n")
                            result = block.get("content", "")
                            if len(str(result)) > 500:
                                result = str(result)[:500] + "..."
                            lines.append(f"```\n{result}\n```")
                            lines.append("")
            else:
                lines.append(f"## {role.title()}\n")
                lines.append(str(content))
                lines.append("")

        return "\n".join(lines)
