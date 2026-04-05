"""
Trajectory Recorder — Records agent trajectories for reinforcement learning.

Captures complete turn-level data including user messages, model responses,
tool calls, tool results, timing, and token usage. Supports JSON-based
persistence and replay capability for training data collection.

Usage
-----
    recorder = TrajectoryRecorder(session_id="abc123")
    recorder.start_turn(user_message="Fix the bug in main.py")
    recorder.add_tool_call("read_file", {"path": "main.py"}, "tool_001")
    recorder.add_tool_result("tool_001", "def main(): ...", is_error=False)
    recorder.end_turn(
        model_response="I found the bug...",
        input_tokens=1500,
        output_tokens=500,
        duration_ms=2300,
    )
    await recorder.save()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes.constants import HERMES_DATA_HOME, TRAJECTORY_DIR

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    """Record of a single tool call within a turn."""

    tool_name: str
    tool_input: Dict[str, Any]
    tool_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ToolResultRecord:
    """Record of a tool result within a turn."""

    tool_id: str
    tool_name: str
    result: str
    is_error: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TrajectoryTurn:
    """
    A single conversation turn — one user message + full agent response cycle.

    Attributes
    ----------
    turn_number:
        1-indexed turn number within the session.
    user_message:
        The user's input message.
    model_response:
        The agent's text response (may include tool calls).
    tool_calls:
        List of tool calls made during this turn.
    tool_results:
        List of tool results received during this turn.
    input_tokens:
        Token count for the input (prompt + history).
    output_tokens:
        Token count for the output (response).
    duration_ms:
        Wall-clock duration of the turn in milliseconds.
    model_name:
        The model used for this turn.
    cost_usd:
        Estimated cost for this turn.
    metadata:
        Additional metadata (error info, iteration count, etc.).
    start_time:
        ISO-8601 timestamp when the turn started.
    end_time:
        ISO-8601 timestamp when the turn ended.
    """

    turn_number: int
    user_message: str = ""
    model_response: str = ""
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    tool_results: List[ToolResultRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    model_name: str = ""
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "turn_number": self.turn_number,
            "user_message": self.user_message,
            "model_response": self.model_response,
            "tool_calls": [asdict(tc) for tc in self.tool_calls],
            "tool_results": [asdict(tr) for tr in self.tool_results],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_ms": round(self.duration_ms, 2),
            "model_name": self.model_name,
            "cost_usd": round(self.cost_usd, 8),
            "metadata": self.metadata,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class Trajectory:
    """
    A complete session trajectory — a sequence of turns.

    Attributes
    ----------
    session_id:
        Unique identifier for the session.
    turns:
        Ordered list of conversation turns.
    model_name:
        The primary model used in this session.
    total_input_tokens:
        Cumulative input tokens across all turns.
    total_output_tokens:
        Cumulative output tokens across all turns.
    total_cost_usd:
        Cumulative cost across all turns.
    total_duration_ms:
        Wall-clock duration of the entire session.
    created_at:
        Session start timestamp.
    updated_at:
        Last activity timestamp.
    metadata:
        Additional session metadata.
    """

    session_id: str
    turns: List[TrajectoryTurn] = field(default_factory=list)
    model_name: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "turns": [t.to_dict() for t in self.turns],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "turn_count": len(self.turns),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────────────────────────────────────
# TrajectoryRecorder
# ──────────────────────────────────────────────────────────────────────────────

class TrajectoryRecorder:
    """
    Records agent trajectories for RL training data collection.

    Provides a turn-based recording interface that captures all information
    needed for training data: user input, model output, tool calls, results,
    timing, and token usage. Supports persistence to JSON and replay.

    Parameters
    ----------
    session_id:
        Unique identifier for this recording session.
    output_dir:
        Directory where trajectory files are saved. If ``None``, uses
        the default Hermes trajectory directory.
    model_name:
        The model being used in this session.
    metadata:
        Additional metadata to attach to the trajectory.
    """

    def __init__(
        self,
        session_id: str,
        output_dir: Optional[str] = None,
        model_name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._output_dir = Path(output_dir) if output_dir else HERMES_DATA_HOME / TRAJECTORY_DIR
        self._trajectory = Trajectory(
            session_id=session_id,
            model_name=model_name,
            metadata=metadata or {},
        )
        self._current_turn: Optional[TrajectoryTurn] = None
        self._turn_start_time: float = 0.0
        self._auto_save: bool = True

    @property
    def trajectory(self) -> Trajectory:
        """Access the current trajectory object."""
        return self._trajectory

    @property
    def turn_count(self) -> int:
        """Number of completed turns."""
        return len(self._trajectory.turns)

    @property
    def is_recording(self) -> bool:
        """Whether a turn is currently being recorded."""
        return self._current_turn is not None

    # ── Turn lifecycle ────────────────────────────────────────────────────

    def start_turn(self, user_message: str) -> None:
        """
        Start recording a new conversation turn.

        Parameters
        ----------
        user_message:
            The user's input message.
        """
        if self._current_turn is not None:
            logger.warning("Starting new turn while previous turn is still recording. Ending previous turn.")
            self.end_turn()

        self._turn_start_time = time.monotonic()
        turn_number = len(self._trajectory.turns) + 1

        self._current_turn = TrajectoryTurn(
            turn_number=turn_number,
            user_message=user_message,
        )

    def add_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_id: str,
    ) -> None:
        """
        Record a tool call in the current turn.

        Parameters
        ----------
        tool_name:
            Name of the tool being called.
        tool_input:
            Input parameters for the tool.
        tool_id:
            Unique identifier for this tool call.
        """
        if self._current_turn is None:
            logger.warning("No active turn for tool call recording")
            return

        self._current_turn.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_id=tool_id,
            )
        )

    def add_tool_result(
        self,
        tool_id: str,
        tool_name: str,
        result: str,
        is_error: bool = False,
    ) -> None:
        """
        Record a tool result in the current turn.

        Parameters
        ----------
        tool_id:
            ID matching the corresponding tool call.
        tool_name:
            Name of the tool that produced the result.
        result:
            The tool's output.
        is_error:
            Whether the tool call resulted in an error.
        """
        if self._current_turn is None:
            logger.warning("No active turn for tool result recording")
            return

        self._current_turn.tool_results.append(
            ToolResultRecord(
                tool_id=tool_id,
                tool_name=tool_name,
                result=result[:50_000],  # Truncate very large results
                is_error=is_error,
            )
        )

    def set_model_response(self, response: str) -> None:
        """
        Set the model's text response for the current turn.

        Parameters
        ----------
        response:
            The model's text output.
        """
        if self._current_turn is None:
            logger.warning("No active turn for model response")
            return
        self._current_turn.model_response = response

    def end_turn(
        self,
        model_response: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model_name: Optional[str] = None,
        cost_usd: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        End the current turn and finalize it.

        Parameters
        ----------
        model_response:
            Override model response text.
        input_tokens:
            Input token count for this turn.
        output_tokens:
            Output token count for this turn.
        model_name:
            The model used for this turn.
        cost_usd:
            Estimated cost for this turn.
        metadata:
            Additional turn metadata.
        """
        if self._current_turn is None:
            return

        # Calculate duration
        duration = (time.monotonic() - self._turn_start_time) * 1000  # ms

        # Apply overrides
        if model_response is not None:
            self._current_turn.model_response = model_response
        self._current_turn.input_tokens = input_tokens
        self._current_turn.output_tokens = output_tokens
        self._current_turn.duration_ms = duration
        self._current_turn.model_name = model_name or self._trajectory.model_name
        self._current_turn.cost_usd = cost_usd
        if metadata:
            self._current_turn.metadata.update(metadata)
        self._current_turn.end_time = datetime.now(timezone.utc).isoformat()

        # Add to trajectory
        self._trajectory.turns.append(self._current_turn)

        # Update totals
        self._trajectory.total_input_tokens += input_tokens
        self._trajectory.total_output_tokens += output_tokens
        self._trajectory.total_cost_usd += cost_usd
        self._trajectory.total_duration_ms += duration
        self._trajectory.updated_at = datetime.now(timezone.utc).isoformat()

        self._current_turn = None
        self._turn_start_time = 0.0

    # ── Persistence ───────────────────────────────────────────────────────

    async def save(self) -> Path:
        """
        Save the trajectory to a JSON file.

        Returns
        -------
        Path
            Path to the saved trajectory file.
        """
        import asyncio

        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"trajectory_{self._trajectory.session_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self._output_dir / filename

        data = self._trajectory.to_dict()

        def _write():
            filepath.write_text(
                json.dumps(data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write)

        logger.info(
            "Trajectory saved: %d turns, %d tokens, $%.4f",
            len(self._trajectory.turns),
            self._trajectory.total_input_tokens + self._trajectory.total_output_tokens,
            self._trajectory.total_cost_usd,
        )
        return filepath

    @classmethod
    async def load(cls, filepath: str) -> TrajectoryRecorder:
        """
        Load a trajectory from a JSON file.

        Parameters
        ----------
        filepath:
            Path to the trajectory JSON file.

        Returns
        -------
        TrajectoryRecorder
            A new recorder instance with the loaded trajectory.
        """
        import asyncio

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filepath}")

        def _read():
            return json.loads(path.read_text(encoding="utf-8"))

        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _read)

        # Reconstruct trajectory
        trajectory = Trajectory(
            session_id=data["session_id"],
            model_name=data.get("model_name", ""),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            metadata=data.get("metadata", {}),
        )

        for turn_data in data.get("turns", []):
            tool_calls = [
                ToolCallRecord(**tc) for tc in turn_data.get("tool_calls", [])
            ]
            tool_results = [
                ToolResultRecord(**tr) for tr in turn_data.get("tool_results", [])
            ]
            turn = TrajectoryTurn(
                turn_number=turn_data.get("turn_number", 0),
                user_message=turn_data.get("user_message", ""),
                model_response=turn_data.get("model_response", ""),
                tool_calls=tool_calls,
                tool_results=tool_results,
                input_tokens=turn_data.get("input_tokens", 0),
                output_tokens=turn_data.get("output_tokens", 0),
                duration_ms=turn_data.get("duration_ms", 0.0),
                model_name=turn_data.get("model_name", ""),
                cost_usd=turn_data.get("cost_usd", 0.0),
                metadata=turn_data.get("metadata", {}),
                start_time=turn_data.get("start_time", ""),
                end_time=turn_data.get("end_time", ""),
            )
            trajectory.turns.append(turn)

        recorder = cls(session_id=data["session_id"])
        recorder._trajectory = trajectory
        return recorder

    # ── Replay ────────────────────────────────────────────────────────────

    def replay(self) -> List[Dict[str, Any]]:
        """
        Generate a replay sequence from recorded turns.

        Returns
        -------
        list[dict]
            A list of events in chronological order, each with ``type``,
            ``turn``, and ``data`` keys. Event types: ``"user_message"``,
            ``"tool_call"``, ``"tool_result"``, ``"model_response"``.
        """
        events: List[Dict[str, Any]] = []
        for turn in self._trajectory.turns:
            events.append({
                "type": "user_message",
                "turn": turn.turn_number,
                "data": turn.user_message,
            })
            for tc in turn.tool_calls:
                events.append({
                    "type": "tool_call",
                    "turn": turn.turn_number,
                    "data": {"tool_name": tc.tool_name, "tool_input": tc.tool_input, "tool_id": tc.tool_id},
                })
            for tr in turn.tool_results:
                events.append({
                    "type": "tool_result",
                    "turn": turn.turn_number,
                    "data": {"tool_name": tr.tool_name, "result": tr.result, "is_error": tr.is_error},
                })
            if turn.model_response:
                events.append({
                    "type": "model_response",
                    "turn": turn.turn_number,
                    "data": turn.model_response,
                })
        return events

    # ── Summary ───────────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the recorded trajectory."""
        t = self._trajectory
        tool_counts: Dict[str, int] = {}
        for turn in t.turns:
            for tc in turn.tool_calls:
                tool_counts[tc.tool_name] = tool_counts.get(tc.tool_name, 0) + 1

        return {
            "session_id": t.session_id,
            "turn_count": len(t.turns),
            "total_input_tokens": t.total_input_tokens,
            "total_output_tokens": t.total_output_tokens,
            "total_tokens": t.total_input_tokens + t.total_output_tokens,
            "total_cost_usd": round(t.total_cost_usd, 6),
            "total_duration_ms": round(t.total_duration_ms, 1),
            "avg_tokens_per_turn": (
                (t.total_input_tokens + t.total_output_tokens) // max(1, len(t.turns))
            ),
            "tool_usage_counts": tool_counts,
            "unique_tools_used": len(tool_counts),
            "created_at": t.created_at,
            "updated_at": t.updated_at,
        }
