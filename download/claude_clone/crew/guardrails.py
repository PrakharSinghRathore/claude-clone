"""
Output guardrail system for crew tasks.

Guardrails validate (and optionally correct) task outputs before they are
accepted. They can be simple callable validators, or LLM-powered validators
that understand semantic correctness.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Guardrail result
# ──────────────────────────────────────────────

class GuardrailResult(BaseModel):
    """
    Result of a guardrail validation check.

    Attributes:
        passed:          Whether the output passed the guardrail check.
        reason:          Human-readable explanation of why the output
                         passed or failed.
        corrected_output: If the guardrail can fix the output, this
                         field contains the corrected version. ``None``
                         means no correction was attempted or possible.
    """

    passed: bool
    reason: str
    corrected_output: Optional[str] = None

    model_config = {"frozen": False}


# ──────────────────────────────────────────────
# Abstract base guardrail
# ──────────────────────────────────────────────

class BaseGuardrail(ABC):
    """
    Abstract base class for all guardrails.

    Subclasses must implement :meth:`validate`, which receives the raw
    task output string and returns a :class:`GuardrailResult`.
    """

    @abstractmethod
    def validate(self, output: str, agent_role: str = "") -> GuardrailResult:
        """
        Validate a task output.

        Args:
            output: The raw string output produced by an agent.
            agent_role: The role of the agent that produced the output
                (useful for context-aware validation).

        Returns:
            A :class:`GuardrailResult` indicating pass/fail and an
            optional corrected output.
        """
        ...


# ──────────────────────────────────────────────
# LLM-powered guardrail
# ──────────────────────────────────────────────

class LLMGuardrail(BaseGuardrail):
    """
    Guardrail that validates task output using an LLM call.

    The LLM is asked to judge whether the output meets the specified
    criteria and, optionally, to produce a corrected version.

    Args:
        validation_prompt: A system prompt that instructs the LLM how
            to validate outputs. Should describe the criteria clearly.
        llm_model: Model identifier string (e.g. ``"anthropic/claude-sonnet-4-20250514"``).
        api_key: Optional API key. Falls back to environment variables.
        attempt_correction: If ``True``, ask the LLM to also produce
            a corrected output when validation fails.

    Example::

        guardrail = LLMGuardrail(
            validation_prompt="Check that the output contains only factual statements "
                            "and does not include any speculative content.",
            llm_model="anthropic/claude-sonnet-4-20250514",
        )
    """

    def __init__(
        self,
        validation_prompt: str,
        llm_model: str = "anthropic/claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        attempt_correction: bool = False,
    ) -> None:
        self.validation_prompt = validation_prompt
        self.llm_model = llm_model
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.attempt_correction = attempt_correction

    def validate(self, output: str, agent_role: str = "") -> GuardrailResult:
        """
        Validate output by asking an LLM.

        This method calls the LLM synchronously. For async contexts,
        consider running it in an executor.

        Args:
            output: The raw task output to validate.
            agent_role: Role of the producing agent.

        Returns:
            A :class:`GuardrailResult` based on the LLM's judgment.
        """
        import anthropic

        judge_prompt = (
            f"{self.validation_prompt}\n\n"
            f"Agent role: {agent_role}\n\n"
            f"Output to validate:\n---\n{output}\n---\n\n"
            "Respond in JSON with keys: "
            '"passed" (boolean), "reason" (string)'
        )
        if self.attempt_correction:
            judge_prompt += (
                ', "corrected_output" (string, only if passed is false)'
            )

        try:
            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            # Detect OpenRouter vs direct Anthropic
            base_url = os.environ.get("OPENROUTER_BASE_URL")
            if base_url and os.environ.get("OPENROUTER_API_KEY"):
                client_kwargs["base_url"] = base_url
                client_kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/claude-clone",
                    "X-Title": "Claude Clone",
                }
            else:
                # If the model has a provider prefix, route through OpenRouter
                if "/" in self.llm_model and "anthropic" not in self.llm_model:
                    client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
                    client_kwargs["default_headers"] = {
                        "HTTP-Referer": "https://github.com/claude-clone",
                        "X-Title": "Claude Clone",
                    }

            client = anthropic.Anthropic(**client_kwargs)

            # Strip provider prefix for Anthropic native
            model = self.llm_model.split("/")[-1] if "/" in self.llm_model else self.llm_model
            if client_kwargs.get("base_url") and "openrouter" not in str(client_kwargs.get("base_url", "")):
                # Direct Anthropic
                pass
            elif client_kwargs.get("base_url"):
                model = self.llm_model  # Keep full name for OpenRouter
            else:
                model = self.llm_model.split("/")[-1]

            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": judge_prompt}],
            )

            text = response.content[0].text.strip() if response.content else ""

            # Try to parse JSON from the response
            json_match = text
            # Handle markdown code blocks
            if "```" in text:
                lines = text.split("\n")
                in_block = False
                json_lines = []
                for line in lines:
                    if line.strip().startswith("```"):
                        if in_block:
                            break
                        in_block = True
                        continue
                    if in_block:
                        json_lines.append(line)
                if json_lines:
                    json_match = "\n".join(json_lines)

            result = json.loads(json_match)
            return GuardrailResult(
                passed=bool(result.get("passed", False)),
                reason=result.get("reason", "LLM validation completed"),
                corrected_output=result.get("corrected_output"),
            )

        except json.JSONDecodeError as e:
            logger.warning("LLM guardrail returned invalid JSON: %s", e)
            return GuardrailResult(
                passed=False,
                reason=f"LLM guardrail returned invalid JSON: {e}",
            )
        except Exception as e:
            logger.error("LLM guardrail call failed: %s", e)
            return GuardrailResult(
                passed=False,
                reason=f"LLM guardrail error: {e}",
            )


# ──────────────────────────────────────────────
# Hallucination guardrail
# ──────────────────────────────────────────────

class HallucinationGuardrail(BaseGuardrail):
    """
    Guardrail that checks for potential hallucinations in the output.

    Uses an LLM to compare the output against the task description and
    context, flagging statements that appear fabricated or unsupported.

    Args:
        llm_model: Model identifier for the checking LLM.
        api_key: Optional API key.
        strict_mode: If ``True``, any detected hallucination causes
            failure. If ``False``, only high-confidence hallucinations
            fail.
    """

    def __init__(
        self,
        llm_model: str = "anthropic/claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        strict_mode: bool = False,
    ) -> None:
        self.llm_model = llm_model
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.strict_mode = strict_mode

    def validate(self, output: str, agent_role: str = "") -> GuardrailResult:
        """
        Check the output for hallucinated or unsupported claims.

        Args:
            output: The raw task output to check.
            agent_role: Role of the producing agent.

        Returns:
            A :class:`GuardrailResult` indicating whether hallucinations
            were detected.
        """
        import anthropic

        prompt = (
            "You are a factual accuracy checker. Review the following output "
            "for potential hallucinations — claims that appear fabricated, "
            "unsupported, or contradicted by common knowledge.\n\n"
            "Consider the following levels:\n"
            "- HIGH: Definitely fabricated (e.g., fake citations, wrong facts)\n"
            "- LOW: Possibly uncertain or speculative but plausible\n\n"
            f"Strict mode: {'ON — any detected issue fails' if self.strict_mode else 'OFF — only HIGH issues fail'}\n\n"
            f"Agent role: {agent_role}\n\n"
            f"Output to check:\n---\n{output}\n---\n\n"
            "Respond in JSON with keys: "
            '"passed" (boolean), "reason" (string), '
            '"hallucinations" (list of objects with "claim" and "severity")'
        )

        try:
            client_kwargs: dict[str, Any] = {"api_key": self.api_key}
            base_url = os.environ.get("OPENROUTER_BASE_URL")
            if base_url and os.environ.get("OPENROUTER_API_KEY"):
                client_kwargs["base_url"] = base_url
                client_kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/claude-clone",
                    "X-Title": "Claude Clone",
                }

            client = anthropic.Anthropic(**client_kwargs)

            model = self.llm_model
            if "/" in model and not client_kwargs.get("base_url"):
                model = model.split("/")[-1]

            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip() if response.content else ""
            if "```" in text:
                lines = text.split("\n")
                in_block = False
                json_lines = []
                for line in lines:
                    if line.strip().startswith("```"):
                        if in_block:
                            break
                        in_block = True
                        continue
                    if in_block:
                        json_lines.append(line)
                if json_lines:
                    text = "\n".join(json_lines)

            result = json.loads(text)
            passed = bool(result.get("passed", True))
            hallucinations = result.get("hallucinations", [])

            # In non-strict mode, only HIGH severity fails
            if not self.strict_mode and not passed:
                has_high = any(
                    h.get("severity", "").upper() == "HIGH"
                    for h in hallucinations
                )
                if not has_high:
                    passed = True

            return GuardrailResult(
                passed=passed,
                reason=result.get("reason", f"Found {len(hallucinations)} potential issues"),
            )

        except Exception as e:
            logger.error("Hallucination guardrail failed: %s", e)
            # Fail open: don't block on guardrail errors
            return GuardrailResult(
                passed=True,
                reason=f"Hallucination check skipped due to error: {e}",
            )


# ──────────────────────────────────────────────
# Callable guardrail wrapper
# ──────────────────────────────────────────────

class CallableGuardrail(BaseGuardrail):
    """
    Wrapper that turns a simple callable into a :class:`BaseGuardrail`.

    The callable should accept a string (the output) and return either:
    - ``True`` / ``False`` (pass / fail),
    - a ``str`` (fail reason), or
    - a :class:`GuardrailResult`.

    Args:
        fn: The validation function.
        name: Optional name for logging purposes.

    Example::

        def no_profanity(output: str) -> bool:
            return "damn" not in output.lower()

        guardrail = CallableGuardrail(fn=no_profanity, name="profanity_check")
    """

    def __init__(self, fn: Callable[[str], Any], name: str = "callable") -> None:
        self.fn = fn
        self.name = name

    def validate(self, output: str, agent_role: str = "") -> GuardrailResult:
        try:
            result = self.fn(output)
            if isinstance(result, GuardrailResult):
                return result
            if isinstance(result, bool):
                return GuardrailResult(
                    passed=result,
                    reason=f"Callable guardrail '{self.name}' {'passed' if result else 'failed'}",
                )
            if isinstance(result, str):
                return GuardrailResult(passed=False, reason=result)
            return GuardrailResult(
                passed=False,
                reason=f"Unexpected return type from guardrail: {type(result)}",
            )
        except Exception as e:
            logger.error("Callable guardrail '%s' error: %s", self.name, e)
            return GuardrailResult(
                passed=False,
                reason=f"Guardrail '{self.name}' raised {type(e).__name__}: {e}",
            )


# ──────────────────────────────────────────────
# Public helper
# ──────────────────────────────────────────────

def process_guardrail(
    guardrail: Union[str, Callable, BaseGuardrail, None],
    output: str,
    agent_role: str = "",
) -> GuardrailResult:
    """
    Process a guardrail of any supported type against an output.

    This helper normalises the ``guardrail`` argument (which may be a
    string prompt, a callable, a :class:`BaseGuardrail` instance, or
    ``None``) and runs the validation.

    Args:
        guardrail: The guardrail to process. If ``None``, the output
            is automatically considered as having passed.
        output: The raw task output to validate.
        agent_role: Role of the producing agent.

    Returns:
        A :class:`GuardrailResult`.
    """
    if guardrail is None:
        return GuardrailResult(passed=True, reason="No guardrail configured")

    if isinstance(guardrail, BaseGuardrail):
        return guardrail.validate(output, agent_role)

    if isinstance(guardrail, str):
        # Treat as an LLM validation prompt
        llm_guard = LLMGuardrail(validation_prompt=guardrail)
        return llm_guard.validate(output, agent_role)

    if callable(guardrail):
        wrapper = CallableGuardrail(fn=guardrail, name=getattr(guardrail, "__name__", "callable"))
        return wrapper.validate(output, agent_role)

    return GuardrailResult(
        passed=False,
        reason=f"Unsupported guardrail type: {type(guardrail)}",
    )
