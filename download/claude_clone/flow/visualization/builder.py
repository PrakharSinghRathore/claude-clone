"""
Build flow graph from Flow class introspection.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Optional

from flow.visualization.schema import (
    EdgeType,
    FlowEdge,
    FlowGraph,
    FlowNode,
    NodeType,
)

if TYPE_CHECKING:
    from flow.flow import Flow

logger = logging.getLogger(__name__)


def build_flow_graph(flow_instance: "Flow") -> FlowGraph:
    """
    Analyze a Flow instance and build its graph representation.

    Inspects the flow's methods for decorators and connections
    to produce a FlowGraph suitable for visualization.

    Args:
        flow_instance: A Flow class instance.

    Returns:
        A FlowGraph representing the flow structure.
    """
    graph = FlowGraph(name=flow_instance.__class__.__name__)

    start_methods = []
    router_methods = []
    and_methods = []
    or_methods = []
    regular_methods = []

    for name, method in inspect.getmembers(flow_instance, predicate=inspect.isfunction):
        if hasattr(method, "_flow_start"):
            start_methods.append((name, method))
        elif hasattr(method, "_flow_router"):
            router_methods.append((name, method))
        elif hasattr(method, "_flow_and"):
            and_methods.append((name, method))
        elif hasattr(method, "_flow_or"):
            or_methods.append((name, method))
        elif hasattr(method, "_flow_listen"):
            regular_methods.append((name, method))

    # Add start nodes
    for name, method in start_methods:
        graph.add_node(FlowNode(
            name=name,
            node_type=NodeType.START,
            method_name=name,
            metadata={"description": method.__doc__ or ""},
        ))

    # Add method nodes
    for name, method in regular_methods:
        graph.add_node(FlowNode(
            name=name,
            node_type=NodeType.METHOD,
            method_name=name,
            metadata={"description": method.__doc__ or ""},
        ))

    # Add router nodes
    for name, method in router_methods:
        graph.add_node(FlowNode(
            name=name,
            node_type=NodeType.ROUTER,
            method_name=name,
            metadata={
                "description": method.__doc__ or "",
                "routes": getattr(method, "_flow_router", []),
            },
        ))

    # Add AND nodes
    for name, method in and_methods:
        graph.add_node(FlowNode(
            name=name,
            node_type=NodeType.AND,
            method_name=name,
        ))

    # Add OR nodes
    for name, method in or_methods:
        graph.add_node(FlowNode(
            name=name,
            node_type=NodeType.OR,
            method_name=name,
        ))

    # Build edges from listen relationships
    for name, method in regular_methods:
        listens_to = getattr(method, "_flow_listen", [])
        for parent in listens_to:
            graph.add_edge(FlowEdge(
                source=parent,
                target=name,
                edge_type=EdgeType.LISTEN,
            ))

    # Build edges from router relationships
    for name, method in router_methods:
        routes = getattr(method, "_flow_router", [])
        for route_target in routes:
            graph.add_edge(FlowEdge(
                source=name,
                target=route_target,
                edge_type=EdgeType.ROUTE,
                condition=f"route to {route_target}",
            ))

    # Build edges from AND/OR branches
    for name, method in and_methods:
        listens_to = getattr(method, "_flow_and", [])
        for parent in listens_to:
            graph.add_edge(FlowEdge(
                source=parent,
                target=name,
                edge_type=EdgeType.AND_BRANCH,
            ))

    for name, method in or_methods:
        listens_to = getattr(method, "_flow_or", [])
        for parent in listens_to:
            graph.add_edge(FlowEdge(
                source=parent,
                target=name,
                edge_type=EdgeType.OR_BRANCH,
            ))

    return graph
