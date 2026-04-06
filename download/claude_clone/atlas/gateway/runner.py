"""
GatewayRunner — Main orchestrator for the Atlas Gateway.

Manages platform adapter lifecycle, session management across platforms,
message routing between platforms, health monitoring with auto-restart,
graceful shutdown, and thread pool for concurrent message processing.

Usage::

    config = GatewayConfig.load("gateway.yaml")
    runner = GatewayRunner(config)
    await runner.start()
    # ... gateway runs ...
    await runner.stop()
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import signal
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from atlas.gateway.config import GatewayConfig, PlatformConfig
from atlas.gateway.delivery import DeliveryRouter
from atlas.gateway.hooks import HookSystem, HookType
from atlas.gateway.mirror import MessageMirror
from atlas.gateway.pairing import PairingManager, PairingRole
from atlas.gateway.session import SessionStore, SessionContext, SessionResetPolicy
from atlas.gateway.status import GatewayStatus
from atlas.gateway.stream_consumer import StreamConsumer

logger = logging.getLogger("atlas.gateway.runner")


# ──────────────────────────────────────────────────────────────────────────────
# Incoming Message
# ──────────────────────────────────────────────────────────────────────────────

class IncomingMessage:
    """Represents an incoming message from any platform."""

    def __init__(
        self,
        platform: str,
        chat_id: str,
        user_id: str,
        text: str,
        message_id: Optional[str] = None,
        username: Optional[str] = None,
        reply_to: Optional[str] = None,
        attachments: Optional[List[str]] = None,
        is_command: bool = False,
        is_edit: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.platform = platform
        self.chat_id = chat_id
        self.user_id = user_id
        self.text = text
        self.message_id = message_id
        self.username = username
        self.reply_to = reply_to
        self.attachments = attachments or []
        self.is_command = is_command
        self.is_edit = is_edit
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"<IncomingMessage platform={self.platform!r} user={self.user_id!r} "
            f"text={self.text[:50]!r}>"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Agent Callback
# ──────────────────────────────────────────────────────────────────────────────

class AgentCallback:
    """Callback interface for processing messages through the agent."""

    def __init__(
        self,
        process_func: Optional[Callable] = None,
        process_async: Optional[Callable] = None,
    ):
        self._process_func = process_func
        self._process_async = process_async

    async def process(self, message: IncomingMessage, context: Dict[str, Any]) -> str:
        """Process a message through the agent and return the response."""
        if self._process_async:
            return await self._process_async(message, context)
        elif self._process_func:
            return self._process_func(message, context)
        return "Gateway: No agent callback configured."


# ──────────────────────────────────────────────────────────────────────────────
# Gateway Runner
# ──────────────────────────────────────────────────────────────────────────────

class GatewayRunner:
    """
    Main orchestrator for the Atlas Gateway.

    Coordinates all platform adapters, manages sessions, routes messages,
    handles delivery, streaming, authentication, and health monitoring.

    Parameters
    ----------
    config:
        Gateway configuration.
    agent_callback:
        Optional callback for processing messages through an AI agent.
    """

    # Mapping of platform names to adapter module paths
    PLATFORM_ADAPTERS: Dict[str, str] = {
        "telegram": "atlas.gateway.platforms.telegram.TelegramAdapter",
        "discord": "atlas.gateway.platforms.discord.DiscordAdapter",
        "slack": "atlas.gateway.platforms.slack.SlackAdapter",
        "whatsapp": "atlas.gateway.platforms.whatsapp.WhatsAppAdapter",
        "signal": "atlas.gateway.platforms.signal.SignalAdapter",
        "matrix": "atlas.gateway.platforms.matrix.MatrixAdapter",
        "email": "atlas.gateway.platforms.email_platform.EmailAdapter",
        "sms": "atlas.gateway.platforms.sms.SMSAdapter",
        "webhook": "atlas.gateway.platforms.webhook.WebhookAdapter",
        "api": "atlas.gateway.platforms.api_server.APIServerAdapter",
        "dingtalk": "atlas.gateway.platforms.dingtalk.DingTalkAdapter",
        "feishu": "atlas.gateway.platforms.feishu.FeishuAdapter",
        "wecom": "atlas.gateway.platforms.wecom.WeComAdapter",
        "mattermost": "atlas.gateway.platforms.mattermost.MattermostAdapter",
    }

    def __init__(
        self,
        config: GatewayConfig,
        agent_callback: Optional[AgentCallback] = None,
    ):
        self._config = config
        self._agent_callback = agent_callback
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Platform adapters
        self._adapters: Dict[str, Any] = {}
        self._adapter_tasks: Dict[str, asyncio.Task] = {}

        # Core subsystems
        self._session_store = SessionStore(
            persist_path=config.session_persist_path,
            reset_policy=config.session_reset_policy,
            timeout=config.session_timeout,
            max_tokens=config.session_max_tokens,
        )
        self._delivery_router = DeliveryRouter(
            config=config,
            session_store=self._session_store,
        )
        self._stream_consumer = StreamConsumer(
            config=config,
            adapters=self._adapters,
            delivery_router=self._delivery_router,
        )
        self._hook_system = HookSystem(
            hooks_dir=config.hooks_dir,
            enabled=config.hooks_enabled,
        )
        self._pairing_manager = PairingManager(
            secret=config.pairing_secret,
            enabled=config.pairing_enabled,
            admin_ids=self._collect_admin_ids(config),
        )
        self._message_mirror = MessageMirror(
            adapters=self._adapters,
            config=config,
            delivery_router=self._delivery_router,
        )
        self._status = GatewayStatus(
            health_check_interval=config.health_check_interval,
        )

        # Thread pool for concurrent message processing
        self._executor = ThreadPoolExecutor(max_workers=config.worker_threads)

        # Message handler callback
        self._on_message: Optional[Callable] = None

        # Agent event handler
        self._on_agent_event: Optional[Callable] = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def config(self) -> GatewayConfig:
        return self._config

    @property
    def session_store(self) -> SessionStore:
        return self._session_store

    @property
    def delivery_router(self) -> DeliveryRouter:
        return self._delivery_router

    @property
    def stream_consumer(self) -> StreamConsumer:
        return self._stream_consumer

    @property
    def hook_system(self) -> HookSystem:
        return self._hook_system

    @property
    def pairing_manager(self) -> PairingManager:
        return self._pairing_manager

    @property
    def message_mirror(self) -> MessageMirror:
        return self._message_mirror

    @property
    def status(self) -> GatewayStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the gateway and all enabled platform adapters.

        Initializes all subsystems, loads adapters, connects to platforms,
        and begins processing messages.
        """
        logger.info("Atlas Gateway starting...")

        # Execute gateway_start hook
        await self._hook_system.execute(
            HookType.GATEWAY_START,
            {"config": self._config.to_dict()},
        )

        # Initialize session store
        await self._session_store.initialize()

        # Load hook plugins
        await self._hook_system.load_plugins()

        # Load and initialize platform adapters
        enabled_platforms = self._config.get_enabled_platforms()
        for name, platform_config in enabled_platforms.items():
            try:
                adapter = await self._load_adapter(name, platform_config)
                if adapter:
                    self._adapters[name] = adapter
                    self._status.register_platform(name)
                    logger.info("Loaded adapter for platform: %s", name)
            except Exception as e:
                logger.error("Failed to load adapter for %s: %s", name, e)
                self._status.record_error(name, "adapter_load", str(e))

        # Connect all adapters
        for name, adapter in self._adapters.items():
            try:
                await self._connect_adapter(name, adapter)
            except Exception as e:
                logger.error("Failed to connect adapter %s: %s", name, e)
                self._status.record_error(name, "connect", str(e))

        # Start health checks
        await self._status.start_health_checks()

        # Start auto-restart monitoring
        self._restart_task = asyncio.create_task(self._auto_restart_loop())

        self._running = True
        logger.info(
            "Atlas Gateway started with %d platform(s)",
            len(self._adapters),
        )

    async def stop(self) -> None:
        """
        Gracefully stop the gateway.

        Disconnects all adapters, persists sessions, and cleans up resources.
        """
        logger.info("Atlas Gateway stopping...")
        self._running = False

        # Execute gateway_stop hook
        await self._hook_system.execute(
            HookType.GATEWAY_STOP,
            {"uptime_seconds": self._status.get_report()["uptime_seconds"]},
        )

        # Stop health checks
        await self._status.stop_health_checks()

        # Cancel restart task
        restart_task = getattr(self, "_restart_task", None)
        if restart_task and not restart_task.done():
            restart_task.cancel()

        # Cancel all adapter tasks
        for name, task in self._adapter_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        self._adapter_tasks.clear()

        # Close stream consumer
        await self._stream_consumer.close()

        # Disconnect all adapters
        for name, adapter in self._adapters.items():
            try:
                if hasattr(adapter, "disconnect"):
                    await adapter.disconnect()
                self._status.platform_disconnected(name, "Gateway shutdown")
                logger.info("Disconnected adapter: %s", name)
            except Exception as e:
                logger.error("Error disconnecting %s: %s", name, e)

        # Close session store
        await self._session_store.close()

        # Shutdown thread pool
        self._executor.shutdown(wait=False)

        self._shutdown_event.set()
        logger.info("Atlas Gateway stopped.")

    # ── Adapter Management ────────────────────────────────────────────────

    async def _load_adapter(
        self, name: str, config: PlatformConfig,
    ) -> Optional[Any]:
        """Load a platform adapter by name."""
        class_path = self.PLATFORM_ADAPTERS.get(name)
        if class_path is None:
            logger.warning("No adapter registered for platform: %s", name)
            return None

        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            return adapter_class(config)
        except ImportError as e:
            logger.warning(
                "Could not import adapter for %s: %s. "
                "Install the required dependencies.",
                name, e,
            )
            return None
        except Exception as e:
            logger.error("Error loading adapter %s: %s", name, e)
            return None

    async def _connect_adapter(self, name: str, adapter: Any) -> None:
        """Connect a platform adapter and start its message loop."""
        if hasattr(adapter, "connect"):
            await adapter.connect()

        self._status.platform_connected(name)

        # Execute platform_connect hook
        await self._hook_system.execute(
            HookType.PLATFORM_CONNECT,
            {"platform": name},
            platform=name,
        )

        # Start the adapter's message polling loop
        if hasattr(adapter, "get_updates"):
            task = asyncio.create_task(
                self._adapter_message_loop(name, adapter),
                name=f"adapter-{name}",
            )
            self._adapter_tasks[name] = task

    async def _adapter_message_loop(self, name: str, adapter: Any) -> None:
        """Poll for messages from an adapter and route them."""
        logger.info("Starting message loop for %s", name)

        try:
            while self._running:
                try:
                    updates = await adapter.get_updates()
                    if updates:
                        for update in updates:
                            try:
                                await self._handle_incoming_message(update)
                            except Exception as e:
                                logger.error(
                                    "Error handling message from %s: %s",
                                    name, e,
                                )
                                self._status.record_error(
                                    name, "message_handler", str(e)
                                )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(
                        "Error polling updates from %s: %s", name, e
                    )
                    self._status.record_error(name, "poll", str(e))
                    self._status.platform_unhealthy(name, str(e))

                    # Brief pause before retrying
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("Message loop cancelled for %s", name)
        except Exception as e:
            logger.error("Message loop crashed for %s: %s", name, e)
            self._status.platform_disconnected(name, str(e))

    async def _auto_restart_loop(self) -> None:
        """Monitor adapters and auto-restart disconnected ones."""
        try:
            while self._running:
                await asyncio.sleep(30)
                for name, adapter in self._adapters.items():
                    if hasattr(adapter, "is_connected"):
                        try:
                            connected = await adapter.is_connected()
                            if not connected:
                                logger.warning(
                                    "Adapter %s disconnected, attempting restart...",
                                    name,
                                )
                                self._status.platform_disconnected(
                                    name, "Auto-detected disconnect"
                                )
                                try:
                                    if hasattr(adapter, "disconnect"):
                                        await adapter.disconnect()
                                    await self._connect_adapter(name, adapter)
                                    logger.info("Adapter %s reconnected", name)
                                except Exception as e:
                                    logger.error(
                                        "Failed to restart %s: %s", name, e
                                    )
                        except Exception as e:
                            logger.debug(
                                "Health check error for %s: %s", name, e
                            )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Auto-restart loop error: %s", e)

    # ── Message Handling ──────────────────────────────────────────────────

    async def _handle_incoming_message(self, msg: IncomingMessage) -> None:
        """
        Process an incoming message through the full pipeline.

        Pipeline:
        1. Record received message
        2. Check authentication/pairing
        3. Execute pre-message hooks
        4. Check rate limits
        5. Get or create session
        6. Handle commands
        7. Process through agent
        8. Execute post-message hooks
        9. Mirror if configured
        """
        platform = msg.platform
        user_id = msg.user_id
        chat_id = msg.chat_id

        # Record statistics
        self._status.record_message_received(platform)

        # Check authentication
        has_access, role, auth_error = await self._pairing_manager.check_access(
            user_id, platform
        )
        if not has_access:
            logger.warning(
                "Access denied for %s:%s: %s", platform, user_id, auth_error
            )
            if auth_error and msg.text:
                await self._delivery_router.deliver(
                    user_id=user_id, platform=platform, chat_id=chat_id,
                    text=f"⚠ Access denied: {auth_error}",
                )
            return

        # Check rate limit
        rate_limit_remaining = await self._pairing_manager.check_rate_limit(
            user_id, platform
        )
        if rate_limit_remaining is not None:
            logger.warning(
                "Rate limited: %s:%s (try again in %.0fs)",
                platform, user_id, rate_limit_remaining,
            )
            return

        # Build context
        context: Dict[str, Any] = {
            "platform": platform,
            "user_id": user_id,
            "chat_id": chat_id,
            "username": msg.username,
            "role": role.value if role else "user",
            "message_id": msg.message_id,
            "attachments": msg.attachments,
            "is_command": msg.is_command,
            "is_edit": msg.is_edit,
            "metadata": msg.metadata,
        }

        # Pre-message hook
        pre_result = await self._hook_system.execute(
            HookType.PRE_MESSAGE, context, platform=platform,
        )
        if pre_result.abort:
            logger.debug("Message aborted by pre-message hook")
            return

        # Get or create session
        session = await self._session_store.get_or_create(user_id, platform, chat_id)
        self._status.set_active_sessions(len(self._session_store._cache))

        # Add user message to session
        session.add_message("user", msg.text, metadata={"message_id": msg.message_id})

        # Handle commands
        if self._hook_system.is_command(msg.text):
            cmd_response = await self._hook_system.process_command(
                msg.text.strip().split()[0], context,
            )
            if cmd_response:
                await self._delivery_router.deliver(
                    user_id=user_id, platform=platform, chat_id=chat_id,
                    text=cmd_response, reply_to=msg.message_id,
                )
                await self._session_store.save(session)
                return

        # Process through agent
        response_text = ""
        if self._agent_callback:
            try:
                # Check if streaming is supported
                if (self._config.streaming_enabled
                        and platform in self._stream_consumer.EDIT_SUPPORTED_PLATFORMS):
                    response_text = await self._process_with_streaming(
                        msg, session, context
                    )
                else:
                    response_text = await self._agent_callback.process(msg, context)
                    await self._delivery_router.deliver_agent_response(
                        user_id=user_id, platform=platform, chat_id=chat_id,
                        response_text=response_text,
                    )
            except Exception as e:
                logger.error("Agent processing error: %s", e)
                response_text = f"Error: {e}"
                self._status.record_error(platform, "agent", str(e))
        else:
            response_text = "Gateway: No agent configured."

        # Add response to session
        session.add_message("assistant", response_text)

        # Post-message hook
        await self._hook_system.execute(
            HookType.POST_MESSAGE,
            {**context, "response": response_text},
            platform=platform,
        )

        # Save session
        await self._session_store.save(session)

        # Mirror if configured
        if self._config.mirroring_enabled:
            try:
                await self._message_mirror.mirror_message(
                    source_platform=platform,
                    source_chat_id=chat_id,
                    text=msg.text,
                    source_user_id=user_id,
                    source_username=msg.username,
                    source_message_id=msg.message_id,
                    attachments=msg.attachments if msg.attachments else None,
                )
            except Exception as e:
                logger.debug("Mirror error: %s", e)

    async def _process_with_streaming(
        self,
        msg: IncomingMessage,
        session: SessionContext,
        context: Dict[str, Any],
    ) -> str:
        """Process a message with streaming response."""
        stream_id = await self._stream_consumer.start_stream(
            platform=msg.platform,
            chat_id=msg.chat_id,
            user_id=msg.user_id,
        )

        full_response = ""

        try:
            if self._agent_callback and self._agent_callback._process_async:
                # For async callbacks, process and then deliver chunks
                response = await self._agent_callback._process_async(msg, context)
                # Simulate chunked delivery
                chunk_size = self._config.streaming_chunk_size
                for i in range(0, len(response), chunk_size):
                    chunk = response[i:i + chunk_size]
                    await self._stream_consumer.send_chunk(stream_id, chunk)
                    await asyncio.sleep(0.05)
                full_response = response
            else:
                full_response = await self._agent_callback.process(msg, context)
        finally:
            await self._stream_consumer.finish_stream(stream_id)

        return full_response

    # ── Public API ────────────────────────────────────────────────────────

    def set_agent_callback(self, callback: AgentCallback) -> None:
        """Set the agent processing callback."""
        self._agent_callback = callback

    def set_message_handler(self, handler: Callable) -> None:
        """Set a custom message handler (bypasses default pipeline)."""
        self._on_message = handler

    def get_adapter(self, name: str) -> Optional[Any]:
        """Get a platform adapter by name."""
        return self._adapters.get(name)

    def get_adapters(self) -> Dict[str, Any]:
        """Return all loaded platform adapters."""
        return dict(self._adapters)

    async def send_message(
        self,
        platform: str,
        chat_id: str,
        text: str,
        user_id: str = "",
        **kwargs: Any,
    ) -> Any:
        """
        Send a message through a specific platform.

        Convenience method for external integrations.
        """
        return await self._delivery_router.deliver(
            user_id=user_id,
            platform=platform,
            chat_id=chat_id,
            text=text,
            **kwargs,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get the current gateway status report."""
        return self._status.get_report()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _collect_admin_ids(config: GatewayConfig) -> List[str]:
        """Collect admin user IDs from all platform configs."""
        admin_ids: List[str] = []
        for name, pc in config.platforms.items():
            admin_ids.extend(pc.admin_ids)
        return admin_ids
