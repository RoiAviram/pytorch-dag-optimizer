"""
dag_core.py
-----------
Custom ComputationDAG built from scratch — no external graph libraries.

Data structure:
  - Nodes  : dict[node_id -> DAGNode]
  - Edges  : adjacency list  dict[node_id -> list[node_id]]  (directed, u->v means v depends on u)
  - Topo   : Kahn's BFS algorithm (guarantees valid forward-pass ordering)
  - Export : lightweight JSON with metadata, nodes (topo order), and edges
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Node dataclass
# ---------------------------------------------------------------------------

@dataclass
class DAGNode:
    """A single operator node in the computation graph."""
    node_id: str                          # Unique identifier (matches fx node name)
    op_type: str                          # High-level op type: 'placeholder', 'call_module', etc.
    name: str                             # Human-readable name / target string
    output_shape: list[int] | None = None # Shape of the tensor this node produces
    kwargs: dict[str, Any] = field(default_factory=dict)  # Extra metadata

    def to_dict(self) -> dict:
        return {
            "node_id":      self.node_id,
            "op_type":      self.op_type,
            "name":         self.name,
            "output_shape": self.output_shape,
            "kwargs":       self.kwargs,
        }


# ---------------------------------------------------------------------------
# ComputationDAG
# ---------------------------------------------------------------------------

class ComputationDAG:
    """
    Directed Acyclic Graph for neural-network computation graphs.

    Internally stores:
      _nodes : dict[str, DAGNode]         — node registry
      _adj   : dict[str, list[str]]       — adjacency list (u -> successors)
      _radj  : dict[str, list[str]]       — reverse adjacency (v -> predecessors)
    """

    def __init__(self, name: str = "ComputationDAG"):
        self.name: str = name
        self._nodes: dict[str, DAGNode]       = {}
        self._adj:   dict[str, list[str]]     = {}   # forward edges
        self._radj:  dict[str, list[str]]     = {}   # reverse edges (for in-degree)

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_node(self, node: DAGNode) -> None:
        """Register a node. Idempotent if already present."""
        nid = node.node_id
        if nid in self._nodes:
            return
        self._nodes[nid] = node
        self._adj.setdefault(nid, [])
        self._radj.setdefault(nid, [])

    def add_edge(self, src: str, dst: str) -> None:
        """
        Add a directed edge src -> dst.
        Edge means: *dst depends on src* (src must execute before dst).
        Both nodes must be registered first.
        """
        if src not in self._nodes or dst not in self._nodes:
            raise ValueError(
                f"add_edge({src!r}, {dst!r}): both nodes must exist before adding an edge."
            )
        if dst not in self._adj[src]:
            self._adj[src].append(dst)
        if src not in self._radj[dst]:
            self._radj[dst].append(src)

    # ------------------------------------------------------------------
    # Kahn's Topological Sort
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """
        Kahn's BFS-based topological sort.

        Algorithm:
          1. Compute in-degree for every node.
          2. Enqueue nodes with in-degree == 0 (sources / inputs).
          3. While queue non-empty:
               pop u, append to result
               for each successor v of u:
                 decrement in-degree[v]
                 if in-degree[v] == 0: enqueue v
          4. If result length < total nodes → cycle detected.

        Returns:
          List of node_ids in a valid topological (execution) order.

        Raises:
          ValueError: if the graph contains a cycle.
        """
        # Step 1: compute in-degrees
        in_degree: dict[str, int] = {nid: len(preds) for nid, preds in self._radj.items()}

        # Step 2: seed queue with all source nodes
        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)

        topo_order: list[str] = []

        # Step 3: BFS
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            for v in self._adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        # Step 4: cycle detection
        if len(topo_order) != len(self._nodes):
            raise ValueError(
                "Cycle detected in ComputationDAG — this is not a valid DAG."
            )

        return topo_order

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        """
        Serialise the DAG to a lightweight JSON-compatible dict.

        Structure:
          {
            "metadata": { name, num_nodes, num_edges, created_at },
            "nodes": [ DAGNode.to_dict(), ... ],   # topologically sorted
            "edges": [ {"src": ..., "dst": ...}, ... ]
          }
        """
        topo = self.topological_sort()

        # Build node list in topo order
        nodes_json = [self._nodes[nid].to_dict() for nid in topo]

        # Build flat edge list
        edges_json = [
            {"src": src, "dst": dst}
            for src, successors in self._adj.items()
            for dst in successors
        ]

        num_edges = sum(len(v) for v in self._adj.values())

        return {
            "metadata": {
                "graph_name":  self.name,
                "num_nodes":   len(self._nodes),
                "num_edges":   num_edges,
                "created_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "nodes": nodes_json,
            "edges": edges_json,
        }

    def export_json(self, path: str) -> None:
        """Write the DAG JSON to *path*, creating parent directories as needed."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = self.to_json()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[ComputationDAG] Exported {payload['metadata']['num_nodes']} nodes "
              f"and {payload['metadata']['num_edges']} edges → {path}")

    # ------------------------------------------------------------------
    # Helpers / dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._nodes)

    def __repr__(self) -> str:
        num_edges = sum(len(v) for v in self._adj.values())
        return (f"ComputationDAG(name={self.name!r}, "
                f"nodes={len(self._nodes)}, edges={num_edges})")
