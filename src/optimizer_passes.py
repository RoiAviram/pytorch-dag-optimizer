"""
optimizer_passes.py
-------------------
Phase 4: A compiler-style multi-pass optimisation framework for the
PyTorch DAG Computation Graph.

Each pass is a distinct Python class that:
  - Inherits from OptimizerPass
  - Implements run(graph_state) -> PassResult
  - Documents which data structure drives its algorithm

Pass hierarchy:
  OptimizerPass (abstract base)
  ├── ConvBNReLUFusionPass    adjacency-list chain detection   [Phase 2, now a class]
  ├── DeadNodeEliminationPass  Set + BFS Queue (reachability)
  ├── ConvBNFoldingPass        adjacency-list 2-node pattern
  └── MemoryFootprintPass      topological traversal + sorting

PassManager: ordered pipeline of passes, sequential execution.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# ── Re-use GraphState from optimizer_agent ───────────────────────────────────
# (imported at call-site to avoid circular imports)


# ═══════════════════════════════════════════════════════════════════════════════
# Result type
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PassResult:
    """Statistics returned by a single optimisation pass."""
    pass_name:     str
    nodes_before:  int
    edges_before:  int
    nodes_after:   int
    edges_after:   int
    nodes_removed: int = 0
    edges_removed: int = 0
    notes:         list[str] = field(default_factory=list)

    @property
    def delta_nodes(self) -> int:
        return self.nodes_before - self.nodes_after

    @property
    def delta_edges(self) -> int:
        return self.edges_before - self.edges_after


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class OptimizerPass(ABC):
    """
    Abstract base for all optimisation passes.

    Every subclass must implement run(gs) which mutates gs in-place and
    returns a PassResult describing what changed.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, gs: Any) -> PassResult: ...

    def _snapshot(self, gs: Any) -> tuple[int, int]:
        n = len(gs.nodes)
        e = sum(len(v) for v in gs.fwd.values())
        return n, e


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 1 — Conv-BN-ReLU Fusion  (Phase 2 logic, now a proper class)
# ═══════════════════════════════════════════════════════════════════════════════

_RELU_CLASSES = ("ReLU", "ReLU6")


class ConvBNReLUFusionPass(OptimizerPass):
    """
    Find Conv2d → BatchNorm2d → ReLU chains where each intermediate node
    has exactly ONE successor (no branching outputs) and fuse them into a
    single Fused_Conv_BN_ReLU node.

    Data structure: adjacency list
    Algorithm    : linear topo-order scan + single-successor guard
    """

    name = "Conv-BN-ReLU Fusion"

    def run(self, gs: Any) -> PassResult:
        before_n, before_e = self._snapshot(gs)
        chains  = self._find_chains(gs)
        fused   = []
        visited: set[str] = set()

        for conv_id, bn_id, relu_id in chains:
            if any(x in visited for x in (conv_id, bn_id, relu_id)):
                continue
            fid = self._fuse(gs, conv_id, bn_id, relu_id)
            fused.append(fid)
            visited.update({conv_id, bn_id, relu_id})

        after_n, after_e = self._snapshot(gs)
        notes = [f"Fused chain: {fid}" for fid in fused]
        return PassResult(
            pass_name    = self.name,
            nodes_before = before_n, edges_before = before_e,
            nodes_after  = after_n,  edges_after  = after_e,
            nodes_removed= before_n - after_n,
            edges_removed= before_e - after_e,
            notes        = notes,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_chains(self, gs: Any) -> list[tuple[str, str, str]]:
        chains = []
        visited: set[str] = set()
        for nid in [n for n in gs.topo if n in gs.nodes]:
            if nid in visited:
                continue
            if gs.module_class(nid) != "Conv2d":
                continue
            conv_succs = gs.successors(nid)
            if len(conv_succs) != 1:
                continue
            bn_id = conv_succs[0]
            if gs.module_class(bn_id) != "BatchNorm2d":
                continue
            bn_succs = gs.successors(bn_id)
            if len(bn_succs) != 1:
                continue
            relu_id = bn_succs[0]
            if gs.module_class(relu_id) not in _RELU_CLASSES:
                continue
            chains.append((nid, bn_id, relu_id))
            visited.update({nid, bn_id, relu_id})
        return chains

    def _fuse(self, gs: Any, conv_id: str, bn_id: str, relu_id: str) -> str:
        fused_id      = f"fused_cbr_{conv_id}"
        ext_preds     = gs.predecessors(conv_id)
        ext_succs     = gs.successors(relu_id)
        out_shape     = gs.nodes[relu_id].get("output_shape")
        conv_shape    = gs.nodes[conv_id].get("output_shape")

        fused_node = {
            "node_id":      fused_id,
            "op_type":      "fused_op",
            "name":         f"FusedConvBNReLU[{conv_id}+{bn_id}+{relu_id}]",
            "output_shape": out_shape,
            "kwargs": {
                "module_class": "Fused_Conv_BN_ReLU",
                "fused_from":   [conv_id, bn_id, relu_id],
                "conv_shape":   conv_shape,
                "fusion_notes": (
                    "Single-kernel Conv+BN+ReLU. BN weights folded into Conv. "
                    "Activation applied in-register — no extra HBM round-trip."
                ),
            },
        }
        gs._add_node(fused_node)
        for p in ext_preds: gs._add_edge(p, fused_id)
        for s in ext_succs:  gs._add_edge(fused_id, s)
        for nid in (conv_id, bn_id, relu_id):
            gs._remove_node(nid)
        return fused_id


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2 — Dead Node Elimination
# ═══════════════════════════════════════════════════════════════════════════════

class DeadNodeEliminationPass(OptimizerPass):
    """
    Remove nodes that are unreachable from any 'output' node when traversing
    the graph in REVERSE (i.e., nodes whose results are never consumed).

    Data structure: set (visited) + deque (BFS queue)
    Algorithm     : Reverse BFS from every output node.
                    Any node NOT visited is dead → removal candidate.
    """

    name = "Dead Node Elimination"

    def run(self, gs: Any) -> PassResult:
        before_n, before_e = self._snapshot(gs)

        # ── Step 1: seed BFS with all output nodes ────────────────────────────
        queue:   deque[str] = deque()
        visited: set[str]   = set()

        for nid, node in gs.nodes.items():
            if node.get("op_type") == "output":
                queue.append(nid)
                visited.add(nid)

        # ── Step 2: reverse BFS ───────────────────────────────────────────────
        # Follow PREDECESSOR edges (reverse direction) so we walk from outputs
        # back toward inputs, marking every live node.
        while queue:
            current = queue.popleft()
            for pred in gs.predecessors(current):
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)

        # ── Step 3: remove unreachable nodes ──────────────────────────────────
        dead_nodes = [nid for nid in list(gs.nodes) if nid not in visited]
        for nid in dead_nodes:
            gs._remove_node(nid)

        after_n, after_e = self._snapshot(gs)
        notes = ([f"Removed dead node: {n}" for n in dead_nodes]
                 if dead_nodes else
                 ["No dead nodes found — graph is fully connected ✓"])

        return PassResult(
            pass_name    = self.name,
            nodes_before = before_n, edges_before = before_e,
            nodes_after  = after_n,  edges_after  = after_e,
            nodes_removed= len(dead_nodes),
            edges_removed= before_e - after_e,
            notes        = notes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 3 — Conv-BN Folding  (2-node pattern, no ReLU required)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBNFoldingPass(OptimizerPass):
    """
    Find any remaining Conv2d → BatchNorm2d pair (the BN may be followed by
    anything — typically an elementwise 'add' on a residual branch) and
    replace the pair with a single Folded_Conv_BN node.

    This targets the 'conv2' + downsample branches missed by ConvBNReLUFusionPass
    because their BN output feeds a skip-connection 'add' node, not a ReLU.

    Data structure: adjacency list
    Algorithm     : topo-order linear scan; single-predecessor guard on BN
                    to ensure we're folding only a strict 2-node chain.
    """

    name = "Conv-BN Folding"

    def run(self, gs: Any) -> PassResult:
        before_n, before_e = self._snapshot(gs)
        folded: list[str]  = []
        visited: set[str]  = set()

        live_topo = [n for n in gs.topo if n in gs.nodes]

        for conv_id in live_topo:
            if conv_id in visited:
                continue
            if gs.module_class(conv_id) != "Conv2d":
                continue

            conv_succs = gs.successors(conv_id)
            if len(conv_succs) != 1:
                continue

            bn_id = conv_succs[0]
            if bn_id in visited:
                continue
            if gs.module_class(bn_id) != "BatchNorm2d":
                continue

            # Guard: BN must have Conv as its ONLY predecessor (strict 2-chain)
            if len(gs.predecessors(bn_id)) != 1:
                continue

            fid = self._fold(gs, conv_id, bn_id)
            folded.append(fid)
            visited.update({conv_id, bn_id})

        after_n, after_e = self._snapshot(gs)
        return PassResult(
            pass_name    = self.name,
            nodes_before = before_n, edges_before = before_e,
            nodes_after  = after_n,  edges_after  = after_e,
            nodes_removed= before_n - after_n,
            edges_removed= before_e - after_e,
            notes        = [f"Folded Conv-BN: {fid}" for fid in folded],
        )

    def _fold(self, gs: Any, conv_id: str, bn_id: str) -> str:
        folded_id  = f"folded_cb_{conv_id}"
        ext_preds  = gs.predecessors(conv_id)
        ext_succs  = gs.successors(bn_id)
        out_shape  = gs.nodes[bn_id].get("output_shape")

        folded_node = {
            "node_id":      folded_id,
            "op_type":      "folded_op",
            "name":         f"FoldedConvBN[{conv_id}+{bn_id}]",
            "output_shape": out_shape,
            "kwargs": {
                "module_class": "Folded_Conv_BN",
                "fused_from":   [conv_id, bn_id],
                "fold_notes": (
                    "BN scale/bias/mean/var mathematically absorbed into Conv "
                    "weight & bias at inference time. Removes BN forward pass entirely."
                ),
            },
        }
        gs._add_node(folded_node)
        for p in ext_preds: gs._add_edge(p, folded_id)
        for s in ext_succs:  gs._add_edge(folded_id, s)
        for nid in (conv_id, bn_id):
            gs._remove_node(nid)
        return folded_id


# ═══════════════════════════════════════════════════════════════════════════════
# Pass 4 — Memory Footprint Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryFootprintPass(OptimizerPass):
    """
    Annotation-only pass: compute the activation memory produced by every node
    and identify the top-K memory consumers and the estimated peak live memory.

    Data structure: dict (memory table) + sorted list (ranking)
    Algorithm     : topological traversal; for each node compute:
                      memory_bytes = product(output_shape) × 4  [float32]
                    Peak live memory is approximated as the maximum over the
                    topo order of: sum(memory of all nodes currently 'alive').
                    A node is alive from its creation until its last consumer
                    finishes (LRU frontier tracking).

    This pass does NOT remove any nodes — it only annotates them.
    """

    name = "Memory Footprint Analysis"
    TOP_K = 5

    def run(self, gs: Any) -> PassResult:
        before_n, before_e = self._snapshot(gs)

        # ── Build last-use map ────────────────────────────────────────────────
        # For each node, record the last topo position at which it is consumed.
        live_topo = [n for n in gs.topo if n in gs.nodes]
        topo_idx  = {nid: i for i, nid in enumerate(live_topo)}

        last_use: dict[str, int] = {}   # node_id -> last topo index that uses it
        for i, nid in enumerate(live_topo):
            for pred in gs.predecessors(nid):
                # pred is consumed at step i
                last_use[pred] = max(last_use.get(pred, 0), i)
        # Outputs have no consumers — mark as freed after themselves
        for nid in live_topo:
            if nid not in last_use:
                last_use[nid] = topo_idx[nid]

        # ── Memory per node ───────────────────────────────────────────────────
        def node_bytes(nid: str) -> int:
            shape = gs.nodes[nid].get("output_shape")
            if not shape:
                return 0
            return math.prod(shape) * 4   # float32

        # ── Annotate every node with its memory cost ──────────────────────────
        for nid in live_topo:
            mb = node_bytes(nid) / 1e6
            gs.nodes[nid].setdefault("kwargs", {})["memory_mb"] = round(mb, 4)

        # ── Compute peak live memory (sliding frontier) ───────────────────────
        alive: dict[str, int] = {}   # node_id -> bytes currently alive
        peak_bytes  = 0
        peak_set: list[str] = []

        for i, nid in enumerate(live_topo):
            # Activate this node
            alive[nid] = node_bytes(nid)
            # Free nodes whose last use was the PREVIOUS step
            to_free = [n for n, lu in list(alive.items()) if lu < i and n != nid]
            for n in to_free:
                del alive[n]
            # Apply last-use for freed
            alive_filtered = {n: b for n, b in alive.items()
                              if last_use.get(n, 0) >= i}
            total = sum(alive_filtered.values())
            if total > peak_bytes:
                peak_bytes = total
                peak_set   = list(alive_filtered.keys())

        peak_mb = peak_bytes / 1e6

        # ── Rank nodes by memory ──────────────────────────────────────────────
        ranked = sorted(
            [(node_bytes(n) / 1e6, n) for n in live_topo],
            reverse=True
        )
        top_k = ranked[:self.TOP_K]

        notes = [f"Peak live memory: {peak_mb:.2f} MB"]
        notes += [f"Top-{i+1} memory node: {nid} ({mb:.2f} MB)"
                  for i, (mb, nid) in enumerate(top_k)]

        # Store summary on the graph metadata for the frontend
        gs.metadata["memory_analysis"] = {
            "peak_live_mb":     round(peak_mb, 2),
            "peak_live_nodes":  peak_set,
            "top_consumers":    [{"node_id": nid, "memory_mb": round(mb, 2)}
                                 for mb, nid in top_k],
        }

        after_n, after_e = self._snapshot(gs)
        return PassResult(
            pass_name    = self.name,
            nodes_before = before_n, edges_before = before_e,
            nodes_after  = after_n,  edges_after  = after_e,
            nodes_removed= 0,
            edges_removed= 0,
            notes        = notes,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Pass Manager
# ═══════════════════════════════════════════════════════════════════════════════

class PassManager:
    """
    Runs an ordered list of OptimizerPass objects sequentially against a
    GraphState, collecting PassResult from each.

    Data structure: ordered list (pass pipeline)
    Pattern       : Iterator / Chain-of-responsibility
    """

    def __init__(self, passes: list[OptimizerPass]) -> None:
        self._passes: list[OptimizerPass] = passes

    def run(self, gs: Any) -> list[PassResult]:
        results: list[PassResult] = []
        for p in self._passes:
            result = p.run(gs)
            results.append(result)
        return results

    def add_pass(self, p: OptimizerPass) -> None:
        self._passes.append(p)
