"""
Schema definitions for flow visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(str, Enum):
    START = "start"
    METHOD = "method"
    ROUTER = "router"
    AND = "and"
    OR = "or"
    END = "end"


class EdgeType(str, Enum):
    LISTEN = "listen"
    ROUTE = "route"
    AND_BRANCH = "and_branch"
    OR_BRANCH = "or_branch"


@dataclass
class FlowNode:
    """A node in the flow graph."""
    name: str
    node_type: NodeType
    method_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.node_type.value,
            "method": self.method_name,
            "metadata": self.metadata,
        }


@dataclass
class FlowEdge:
    """An edge connecting two flow nodes."""
    source: str
    target: str
    edge_type: EdgeType
    condition: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value,
            "condition": self.condition,
        }


@dataclass
class FlowGraph:
    """A directed graph representation of a flow."""
    name: str
    nodes: List[FlowNode] = field(default_factory=list)
    edges: List[FlowEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: FlowNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: FlowEdge) -> None:
        self.edges.append(edge)

    def get_node(self, name: str) -> Optional[FlowNode]:
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    def get_outgoing_edges(self, node_name: str) -> List[FlowEdge]:
        return [e for e in self.edges if e.source == node_name]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_mermaid(self) -> str:
        """Generate a Mermaid.js flowchart representation."""
        lines = [f"graph TD"]
        node_id_map = {}
        for i, node in enumerate(self.nodes):
            nid = f"N{i}"
            node_id_map[node.name] = nid
            label = node.name.replace("_", " ").title()
            if node.node_type == NodeType.ROUTER:
                lines.append(f"    {nid}{{{label}}}[/Router/]")
            elif node.node_type == NodeType.START:
                lines.append(f"    {nid}(({label}))")
            elif node.node_type == NodeType.AND:
                lines.append(f"    {nid}[/AND/]")
            elif node.node_type == NodeType.OR:
                lines.append(f"    {nid}{{OR}}")
            else:
                lines.append(f"    {nid}[{label}]")
        for edge in self.edges:
            src = node_id_map.get(edge.source, edge.source)
            tgt = node_id_map.get(edge.target, edge.target)
            label = ""
            if edge.condition:
                label = f"|{edge.condition}|"
            if edge.edge_type == EdgeType.OR_BRANCH:
                lines.append(f"    {src} -.-> {tgt} {label}")
            else:
                lines.append(f"    {src} --> {tgt} {label}")
        return "\n".join(lines)
