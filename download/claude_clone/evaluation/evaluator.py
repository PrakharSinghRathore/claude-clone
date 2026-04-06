"""
Agent evaluation framework.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EvaluationMetric:
    """Predefined evaluation metrics."""
    
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    TOOL_EFFICIENCY = "tool_efficiency"
    RESPONSE_TIME = "response_time"
    TOKEN_EFFICIENCY = "token_efficiency"


@dataclass
class EvaluationResult:
    """Result of evaluating an agent's output."""
    agent_role: str = ""
    task_description: str = ""
    metric_name: str = ""
    score: float = 0.0
    max_score: float = 1.0
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def normalized_score(self) -> float:
        """Score normalized to 0-1 range."""
        return self.score / self.max_score if self.max_score > 0 else 0.0
    
    @property
    def percentage(self) -> float:
        """Score as a percentage (0-100)."""
        return self.normalized_score * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_role": self.agent_role,
            "task": self.task_description,
            "metric": self.metric_name,
            "score": self.score,
            "max_score": self.max_score,
            "percentage": self.percentage,
            "details": self.details,
        }


class AgentEvaluator:
    """
    Framework for evaluating agent performance.
    
    Supports built-in metrics and custom evaluation functions.
    
    Args:
        custom_metrics: Optional dictionary of custom metric functions.
            Each function should accept (output: str, expected: str) -> float.
    """
    
    def __init__(self, custom_metrics: Optional[Dict[str, Callable]] = None):
        self.custom_metrics = custom_metrics or {}
    
    def evaluate(
        self,
        output: str,
        expected: str = "",
        metric: str = EvaluationMetric.ACCURACY,
        agent_role: str = "",
        task_description: str = "",
    ) -> EvaluationResult:
        """
        Evaluate an agent's output against expected result.
        
        Args:
            output: The agent's actual output.
            expected: The expected/desired output.
            metric: The metric to use.
            agent_role: The agent's role (for reporting).
            task_description: Description of the task (for reporting).
        
        Returns:
            An EvaluationResult with the score.
        """
        if metric in self.custom_metrics:
            score = self.custom_metrics[metric](output, expected)
            return EvaluationResult(
                agent_role=agent_role,
                task_description=task_description,
                metric_name=metric,
                score=float(score),
                max_score=1.0,
                details="Custom metric evaluation",
            )
        
        if metric == EvaluationMetric.ACCURACY:
            return self._evaluate_accuracy(output, expected, agent_role, task_description)
        elif metric == EvaluationMetric.COMPLETENESS:
            return self._evaluate_completeness(output, expected, agent_role, task_description)
        elif metric == EvaluationMetric.RELEVANCE:
            return self._evaluate_relevance(output, expected, agent_role, task_description)
        elif metric == EvaluationMetric.COHERENCE:
            return self._evaluate_coherence(output, agent_role, task_description)
        else:
            return EvaluationResult(
                agent_role=agent_role,
                task_description=task_description,
                metric_name=metric,
                score=0.0,
                details=f"Unknown metric: {metric}",
            )
    
    def _evaluate_accuracy(
        self, output: str, expected: str, agent_role: str, task: str
    ) -> EvaluationResult:
        """Simple word-overlap accuracy."""
        if not expected:
            return EvaluationResult(agent_role=agent_role, task_description=task, metric_name="accuracy")
        
        output_words = set(re.findall(r'\w+', output.lower()))
        expected_words = set(re.findall(r'\w+', expected.lower()))
        
        if not expected_words:
            return EvaluationResult(agent_role=agent_role, task_description=task, metric_name="accuracy")
        
        overlap = output_words & expected_words
        score = len(overlap) / len(expected_words)
        
        return EvaluationResult(
            agent_role=agent_role,
            task_description=task,
            metric_name="accuracy",
            score=score,
            max_score=1.0,
            details=f"Word overlap: {len(overlap)}/{len(expected_words)} ({score:.1%})",
        )
    
    def _evaluate_completeness(
        self, output: str, expected: str, agent_role: str, task: str
    ) -> EvaluationResult:
        """Check if expected key points are covered."""
        if not expected:
            return EvaluationResult(agent_role=agent_role, task_description=task, metric_name="completeness")
        
        expected_sentences = [s.strip() for s in expected.split(".") if s.strip()]
        if not expected_sentences:
            return EvaluationResult(agent_role=agent_role, task_description=task, metric_name="completeness")
        
        covered = 0
        for sentence in expected_sentences:
            keywords = set(re.findall(r'\w+', sentence.lower())) - {"the", "a", "an", "is", "are", "was", "were"}
            if keywords and any(kw in output.lower() for kw in keywords):
                covered += 1
        
        score = covered / len(expected_sentences)
        return EvaluationResult(
            agent_role=agent_role,
            task_description=task,
            metric_name="completeness",
            score=score,
            max_score=1.0,
            details=f"Covered {covered}/{len(expected_sentences)} expected points",
        )
    
    def _evaluate_relevance(
        self, output: str, expected: str, agent_role: str, task: str
    ) -> EvaluationResult:
        """Check relevance using keyword presence."""
        if not expected:
            return EvaluationResult(agent_role=agent_role, task_description=task, metric_name="relevance")
        
        expected_keywords = set(re.findall(r'\w+', expected.lower()))
        expected_keywords -= {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but"}
        
        if not expected_keywords:
            return EvaluationResult(agent_role=agent_role, task_description=task, metric_name="relevance")
        
        output_lower = output.lower()
        relevant = sum(1 for kw in expected_keywords if kw in output_lower)
        score = relevant / len(expected_keywords)
        
        return EvaluationResult(
            agent_role=agent_role,
            task_description=task,
            metric_name="relevance",
            score=score,
            max_score=1.0,
            details=f"{relevant}/{len(expected_keywords)} keywords present",
        )
    
    def _evaluate_coherence(
        self, output: str, agent_role: str, task: str
    ) -> EvaluationResult:
        """Basic coherence check based on sentence structure."""
        sentences = [s.strip() for s in output.replace("\n", ".").split(".") if s.strip()]
        if not sentences:
            return EvaluationResult(agent_role=agent_role, task_description=task, metric_name="coherence", score=0.0)
        
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        score = min(1.0, avg_length / 15)  # Assume ~15 words per sentence is ideal
        
        return EvaluationResult(
            agent_role=agent_role,
            task_description=task,
            metric_name="coherence",
            score=score,
            max_score=1.0,
            details=f"Average sentence length: {avg_length:.1f} words",
        )
    
    def evaluate_batch(
        self,
        results: List[Dict[str, str]],
        metric: str = EvaluationMetric.ACCURACY,
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of results.
        
        Args:
            results: List of dicts with 'output', 'expected', 'agent_role', 'task' keys.
            metric: The metric to use.
        
        Returns:
            List of EvaluationResult instances.
        """
        return [
            self.evaluate(
                output=r.get("output", ""),
                expected=r.get("expected", ""),
                metric=metric,
                agent_role=r.get("agent_role", ""),
                task_description=r.get("task", ""),
            )
            for r in results
        ]
