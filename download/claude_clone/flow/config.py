"""
Flow configuration utilities.
"""

from typing import Any, Dict, Optional


class FlowConfig:
    """
    Configuration for a Flow instance.

    Attributes:
        name: Human-readable flow name.
        description: Flow description.
        max_iterations: Maximum number of steps to execute.
        store_messages: Whether to store intermediate messages.
        finish_timeout: Timeout in seconds for flow completion.
        extra: Arbitrary extra configuration.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_iterations: int = 100,
        store_messages: bool = True,
        finish_timeout: Optional[int] = None,
        **kwargs: Any,
    ):
        self.name = name
        self.description = description
        self.max_iterations = max_iterations
        self.store_messages = store_messages
        self.finish_timeout = finish_timeout
        self.extra = kwargs

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "description": self.description,
            "max_iterations": self.max_iterations,
            "store_messages": self.store_messages,
            "finish_timeout": self.finish_timeout,
        }
        result.update(self.extra)
        return result


def flow_config(
    cls: Optional[type] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    max_iterations: int = 100,
    **kwargs: Any,
):
    """
    Decorator to attach configuration to a Flow class.

    Usage::

        @flow_config(name="my_flow", max_iterations=50)
        class MyFlow(Flow):
            ...
    """
    def decorator(klass: type) -> type:
        config = FlowConfig(
            name=name or klass.__name__,
            description=description,
            max_iterations=max_iterations,
            **kwargs,
        )
        klass._flow_config = config  # type: ignore[attr-defined]
        return klass

    if cls is not None:
        return decorator(cls)
    return decorator
