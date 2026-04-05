"""
Hermes Mixture of Agents Tool — multi-model consensus for better outputs.

Features:
- Send prompt to multiple AI models
- Aggregate responses
- Voting/consensus mechanism
- Quality scoring
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

from hermes.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

_SUPPORTED_MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-haiku-20240307",
    "anthropic/claude-opus-4-20250514",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-pro-1.5",
    "meta-llama/llama-3-70b-instruct",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def hermes_mixture_of_agents(
    prompt: str,
    models: str = "",
    strategy: str = "majority",
    max_tokens: int = 1000,
) -> str:
    """Send a prompt to multiple AI models and aggregate their responses.

    param prompt (str): — The prompt to send to all models.
    param models (str): — Comma-separated model list. Empty = default set.
    param strategy (str): — Aggregation strategy: majority, best, all. Default: majority.
    param max_tokens (int): — Max tokens per model. Default: 1000.
    """
    # Parse models
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
    else:
        # Default: use 2-3 diverse models
        model_list = [
            "anthropic/claude-sonnet-4-20250514",
            "google/gemini-2.0-flash-exp:free",
            "openai/gpt-4o-mini",
        ]

    if strategy not in ("majority", "best", "all"):
        strategy = "majority"

    # Check for API keys
    api_key = (
        os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
    )
    if not api_key:
        return (
            "Error: No API key found. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY.\n"
            "The mixture of agents tool requires API access to multiple models."
        )

    # Execute queries to each model concurrently
    results = []
    start = time.time()

    for model in model_list:
        try:
            response = await _query_model(model, prompt, api_key, max_tokens)
            results.append({
                "model": model,
                "response": response,
                "success": True,
                "error": None,
                "score": _score_response(response),
            })
        except Exception as e:
            results.append({
                "model": model,
                "response": "",
                "success": False,
                "error": str(e),
                "score": 0,
            })

    elapsed = time.time() - start
    successful = [r for r in results if r["success"]]

    # Format output based on strategy
    lines = [
        f"Mixture of Agents Results ({elapsed:.1f}s)",
        f"Models queried: {len(model_list)}, Successful: {len(successful)}/{len(results)}",
        f"Strategy: {strategy}",
        "",
    ]

    if strategy == "all":
        for r in results:
            status = "OK" if r["success"] else f"ERROR: {r['error']}"
            lines.append(f"--- {r['model']} [{status}] (score: {r['score']}) ---")
            if r["success"]:
                lines.append(r["response"][:500])
            lines.append("")

    elif strategy == "majority" and len(successful) >= 2:
        # Simple majority: find most common themes
        consensus = _find_consensus([r["response"] for r in successful])
        lines.append("=== CONSENSUS ===")
        lines.append(consensus)
        lines.append("")
        lines.append("=== INDIVIDUAL RESPONSES ===")
        for r in successful:
            lines.append(f"\n--- {r['model']} (score: {r['score']}) ---")
            lines.append(r["response"][:300])

    elif strategy == "best" and successful:
        best = max(successful, key=lambda r: r["score"])
        lines.append("=== BEST RESPONSE ===")
        lines.append(f"Model: {best['model']} (score: {best['score']})")
        lines.append("")
        lines.append(best["response"])
        lines.append("")
        lines.append("=== OTHER RESPONSES ===")
        for r in successful:
            if r["model"] != best["model"]:
                lines.append(f"\n--- {r['model']} (score: {r['score']}) ---")
                lines.append(r["response"][:200])

    else:
        # Fallback: just list all
        for r in successful:
            lines.append(f"--- {r['model']} (score: {r['score']}) ---")
            lines.append(r["response"][:400])
            lines.append("")

    return "\n".join(lines)


async def _query_model(
    model: str,
    prompt: str,
    api_key: str,
    max_tokens: int,
) -> str:
    """Query a single model via OpenRouter API."""
    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/claude-clone/hermes",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    url = f"{base_url}/chat/completions"

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _score_response(text: str) -> float:
    """Score a response for quality (heuristic)."""
    if not text:
        return 0.0

    score = 50.0

    # Length: prefer moderate-length responses
    word_count = len(text.split())
    if 20 <= word_count <= 500:
        score += 20
    elif word_count > 500:
        score += 10

    # Structure: prefer responses with lists or sections
    if any(marker in text for marker in ["1.", "-", "*", "•", "First", "Second"]):
        score += 10

    # Clarity: prefer responses without excessive repetition
    sentences = text.split(".")
    unique_sentences = len(set(s.strip().lower() for s in sentences if s.strip()))
    if unique_sentences > len(sentences) * 0.7:
        score += 10

    # Completeness: prefer responses that directly address the prompt
    if any(word in text.lower() for word in ["answer", "solution", "result", "here", "based on"]):
        score += 5

    return min(100, score)


def _find_consensus(responses: List[str]) -> str:
    """Simple consensus extraction from multiple responses."""
    if not responses:
        return "No consensus reached."

    if len(responses) == 1:
        return responses[0]

    # Find common key points by looking for shared sentences/phrases
    import re

    all_sentences = []
    for resp in responses:
        sentences = [s.strip() for s in re.split(r"[.!?\n]", resp) if len(s.strip()) > 15]
        all_sentences.append(sentences)

    # Find sentences that appear in multiple responses
    sentence_counts: Dict[str, int] = {}
    for sentences in all_sentences:
        seen = set()
        for s in sentences:
            s_lower = s.lower()
            for existing in list(sentence_counts.keys()):
                if _similarity(s_lower, existing) > 0.7:
                    sentence_counts[existing] += 1
                    seen.add(s_lower)
                    break
            else:
                if s_lower not in seen:
                    sentence_counts[s_lower] = 1
                    seen.add(s_lower)

    # Get most agreed-upon points
    consensus_points = sorted(sentence_counts.items(), key=lambda x: -x[1])
    top_points = [p[0].capitalize() for p in consensus_points[:5] if p[1] > 1]

    if top_points:
        return "Consensus points agreed upon by multiple models:\n\n" + "\n".join(
            f"- {p}" for p in top_points
        )

    # Fallback: return the highest-scored response
    scored = [(r, _score_response(r)) for r in responses]
    best = max(scored, key=lambda x: x[1])
    return f"No clear consensus. Best individual response (score: {best[1]}):\n\n{best[0]}"


def _similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


async def hermes_model_list() -> str:
    """List supported models for mixture of agents.

    Lists models that can be used with the mixture of agents tool.
    """
    lines = ["Supported models for mixture of agents:\n"]
    categories: Dict[str, List[str]] = {}
    for model in _SUPPORTED_MODELS:
        provider = model.split("/")[0]
        categories.setdefault(provider, []).append(model)

    for provider, models in sorted(categories.items()):
        lines.append(f"  {provider}:")
        for m in models:
            lines.append(f"    - {m}")

    lines.append("\nUsage: Pass comma-separated model names to hermes_mixture_of_agents")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

ToolRegistry.instance().register(
    name="hermes_mixture_of_agents",
    func=hermes_mixture_of_agents,
    description="Send a prompt to multiple AI models and aggregate responses with consensus.",
    toolset="agent",
)

ToolRegistry.instance().register(
    name="hermes_model_list",
    func=hermes_model_list,
    description="List supported models for the mixture of agents tool.",
    toolset="agent",
)
