"""
optimizer_agent.py
------------------
Phase 2: Graph Pattern Matching & Operator Fusion.

Pipeline:
  1. Load output/graph.json → reconstruct in-memory adjacency + reverse-adj
  2. Walk nodes in topological order
  3. Match Conv2d → BatchNorm2d → ReLU chains (strict single-successor guard)
  4. Fuse each chain into a Fused_Conv_BN_ReLU node, rewire external edges
  5. Print professional AI-agent terminal report
  6. Export output/optimized_graph.json

Run:
  python src/optimizer_agent.py
  python src/optimizer_agent.py --input output/graph.json --output output/optimized_graph.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers (graceful fallback if terminal doesn't support them)
# ──────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
DIM    = "\033[2m"


def c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET


# ──────────────────────────────────────────────────────────────────────────────
# In-memory graph representation (rebuilt from JSON)
# ──────────────────────────────────────────────────────────────────────────────

class GraphState:
    """
    Mutable in-memory computation graph rebuilt from graph.json.

    Attributes:
        nodes    : dict[node_id -> node_dict]   (the node payload)
        topo     : list[node_id]                (topological order, maintained)
        fwd      : dict[node_id -> list[node_id]]   forward  adjacency
        rev      : dict[node_id -> list[node_id]]   backward adjacency
    """

    def __init__(self, raw: dict) -> None:
        self.metadata: dict[str, Any] = deepcopy(raw["metadata"])

        # Node registry
        self.nodes: dict[str, dict] = {
            n["node_id"]: deepcopy(n) for n in raw["nodes"]
        }

        # Preserve original topological order  (nodes list IS topo-sorted)
        self.topo: list[str] = [n["node_id"] for n in raw["nodes"]]

        # Build adjacency lists from the flat edge list
        self.fwd: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        self.rev: dict[str, list[str]] = {nid: [] for nid in self.nodes}

        for edge in raw["edges"]:
            s, d = edge["src"], edge["dst"]
            if d not in self.fwd[s]:
                self.fwd[s].append(d)
            if s not in self.rev[d]:
                self.rev[d].append(s)

    # ── helpers ───────────────────────────────────────────────────────────────

    def successors(self, nid: str) -> list[str]:
        return self.fwd.get(nid, [])

    def predecessors(self, nid: str) -> list[str]:
        return self.rev.get(nid, [])

    def module_class(self, nid: str) -> str:
        return self.nodes[nid].get("kwargs", {}).get("module_class", "")

    # ── mutation primitives ───────────────────────────────────────────────────

    def _remove_node(self, nid: str) -> None:
        """Remove node + all its edges from the graph."""
        # detach from successors' rev lists
        for succ in self.fwd.pop(nid, []):
            self.rev[succ] = [p for p in self.rev[succ] if p != nid]
        # detach from predecessors' fwd lists
        for pred in self.rev.pop(nid, []):
            self.fwd[pred] = [s for s in self.fwd[pred] if s != nid]
        self.nodes.pop(nid, None)

    def _add_node(self, node_dict: dict) -> None:
        nid = node_dict["node_id"]
        self.nodes[nid] = node_dict
        self.fwd.setdefault(nid, [])
        self.rev.setdefault(nid, [])

    def _add_edge(self, src: str, dst: str) -> None:
        if dst not in self.fwd[src]:
            self.fwd[src].append(dst)
        if src not in self.rev[dst]:
            self.rev[dst].append(src)

    def to_json(self) -> dict:
        """Serialise back to the same schema as graph.json."""
        # Rebuild topo order: keep original relative ordering, filter to live nodes
        live_set  = set(self.nodes)
        topo_live = [nid for nid in self.topo if nid in live_set]
        # Fused nodes weren't in original topo; append them in stable order
        fused = [nid for nid in self.nodes if nid not in set(topo_live)]
        topo_live.extend(fused)

        nodes_list = [self.nodes[nid] for nid in topo_live]
        edges_list = [
            {"src": s, "dst": d}
            for s, dsts in self.fwd.items()
            for d in dsts
        ]

        num_edges = sum(len(v) for v in self.fwd.values())

        return {
            "metadata": {
                **self.metadata,
                "graph_name":   self.metadata["graph_name"] + " (Optimized)",
                "num_nodes":    len(self.nodes),
                "num_edges":    num_edges,
                "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "nodes": nodes_list,
            "edges": edges_list,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Pattern matching
# ──────────────────────────────────────────────────────────────────────────────

# The module_class values that form the fusable chain
_CONV_CLASS  = "Conv2d"
_BN_CLASS    = "BatchNorm2d"
_RELU_CLASSES = ("ReLU", "ReLU6")    # some variants exist


def find_conv_bn_relu_chains(gs: GraphState) -> list[tuple[str, str, str]]:
    """
    Walk nodes in topological order and find all Conv2d → BN → ReLU chains
    where each intermediate node has EXACTLY ONE successor (no branch outputs).

    Returns:
        List of (conv_id, bn_id, relu_id) tuples — in topo discovery order.
    """
    chains: list[tuple[str, str, str]] = []
    visited: set[str] = set()     # nodes already absorbed into a chain

    # Iterate in topo order so we find chains left-to-right through the graph
    live_topo = [nid for nid in gs.topo if nid in gs.nodes]

    for nid in live_topo:
        if nid in visited:
            continue

        # ── Gate 1: must be a Conv2d ──
        if gs.module_class(nid) != _CONV_CLASS:
            continue

        conv_successors = gs.successors(nid)
        if len(conv_successors) != 1:
            # Conv fans out → skip; can't fuse without changing semantics
            continue

        bn_id = conv_successors[0]
        if gs.module_class(bn_id) not in (_BN_CLASS,):
            continue

        bn_successors = gs.successors(bn_id)
        if len(bn_successors) != 1:
            continue

        relu_id = bn_successors[0]
        if gs.module_class(relu_id) not in _RELU_CLASSES:
            continue

        # ── Valid chain found ──
        chains.append((nid, bn_id, relu_id))
        visited.update({nid, bn_id, relu_id})

    return chains


# ──────────────────────────────────────────────────────────────────────────────
# Graph mutation — fuse one chain
# ──────────────────────────────────────────────────────────────────────────────

def fuse_chain(gs: GraphState, conv_id: str, bn_id: str, relu_id: str) -> str:
    """
    Replace the three-node chain with a single Fused_Conv_BN_ReLU node.

    External wiring:
      - All predecessors of conv_id  →  fused_id
      - fused_id  →  all successors of relu_id

    Returns:
        The new fused node_id.
    """
    fused_id = f"fused_cbr_{conv_id}"

    # Capture external connections BEFORE removing anything
    external_preds = gs.predecessors(conv_id)       # inputs to the chain
    external_succs = gs.successors(relu_id)         # outputs of the chain

    # Shape: inherit relu's output shape (final tensor produced by the chain)
    out_shape = gs.nodes[relu_id].get("output_shape")
    in_shape  = gs.nodes[conv_id].get("output_shape")   # conv's own output (pre-BN)

    fused_node: dict[str, Any] = {
        "node_id":      fused_id,
        "op_type":      "fused_op",
        "name":         f"FusedConvBNReLU[{conv_id}+{bn_id}+{relu_id}]",
        "output_shape": out_shape,
        "kwargs": {
            "module_class":  "Fused_Conv_BN_ReLU",
            "fused_from":    [conv_id, bn_id, relu_id],
            "conv_shape":    in_shape,
            "fusion_notes":  (
                "Single-kernel Conv+BN+ReLU. BatchNorm weights folded into "
                "Conv weights at inference time. Activation applied in-register "
                "without a separate memory round-trip."
            ),
        },
    }

    # 1. Add fused node
    gs._add_node(fused_node)

    # 2. Wire external predecessors → fused
    for pred in external_preds:
        gs._add_edge(pred, fused_id)

    # 3. Wire fused → external successors
    for succ in external_succs:
        gs._add_edge(fused_id, succ)

    # 4. Remove the three original nodes (edges auto-cleaned)
    for nid in (conv_id, bn_id, relu_id):
        gs._remove_node(nid)

    return fused_id


# ──────────────────────────────────────────────────────────────────────────────
# AI-agent terminal report
# ──────────────────────────────────────────────────────────────────────────────

_BANNER_WIDTH = 66

def _banner(title: str) -> str:
    pad = _BANNER_WIDTH - len(title) - 4
    left = pad // 2
    right = pad - left
    return c(f"╔{'═' * (_BANNER_WIDTH - 2)}╗\n"
             f"║  {' ' * left}{title}{' ' * right}  ║\n"
             f"╚{'═' * (_BANNER_WIDTH - 2)}╝", CYAN, BOLD)


def _section(title: str) -> str:
    return c(f"\n  ── {title} {'─' * max(0, 52 - len(title))}", BLUE, BOLD)


def print_report(
    original_nodes:   int,
    original_edges:   int,
    chains:           list[tuple[str, str, str]],
    fused_ids:        list[str],
    optimized_nodes:  int,
    optimized_edges:  int,
    out_path:         str,
) -> None:
    n = len(chains)
    saved_nodes = original_nodes - optimized_nodes
    saved_edges = original_edges - optimized_edges

    # Each Conv→BN→ReLU chain normally needs 2 extra memory round-trips
    # (one between Conv→BN and one between BN→ReLU). Fusion eliminates both.
    rw_eliminated = 2 * n

    print()
    print(_banner("PyTorch DAG Optimizer  ·  AI Analysis Report"))
    print()

    # ── 1. Graph statistics ──────────────────────────────────────────────────
    print(_section("Original Graph Statistics"))
    print(f"    Nodes : {c(original_nodes, YELLOW, BOLD)}")
    print(f"    Edges : {c(original_edges, YELLOW, BOLD)}")
    print()

    # ── 2. Pattern matching results ──────────────────────────────────────────
    print(_section("Pattern Matcher  (Conv2d → BatchNorm2d → ReLU)"))
    if n == 0:
        print(c("    ✗  No fusable chains found.", RED))
    else:
        print(f"    {c(f'✔  {n} fusable chain(s) detected', GREEN, BOLD)}\n")
        for i, (conv_id, bn_id, relu_id) in enumerate(chains, 1):
            fused_id = fused_ids[i - 1]
            print(f"    Chain {c(i, BOLD)}")
            print(f"      {c(conv_id, MAGENTA):30s}  [Conv2d]")
            print(f"        └─▶ {c(bn_id, MAGENTA):26s}  [BatchNorm2d]")
            print(f"              └─▶ {c(relu_id, MAGENTA):22s}  [ReLU]")
            print(f"      {c('▶ Fused into:', DIM)} {c(fused_id, GREEN)}")
            print()

    # ── 3. Fusion actions ────────────────────────────────────────────────────
    print(_section("Graph Mutation Actions"))
    print(f"    Nodes removed  : {c(saved_nodes, RED, BOLD)}  "
          f"({n} × 3 original  →  {n} × 1 fused)")
    print(f"    Edges removed  : {c(saved_edges, RED, BOLD)}")
    print(f"    Optimized graph: {c(optimized_nodes, GREEN, BOLD)} nodes, "
          f"{c(optimized_edges, GREEN, BOLD)} edges")
    print()

    # ── 4. Hardware impact analysis ──────────────────────────────────────────
    print(_section("Hardware Impact  (AI Advisor)"))
    print(f"""
    {c('Operator fusion eliminates intermediate tensor materialisation.', BOLD)}

    In an unfused graph, the output of every operator must be written
    to GPU global memory (HBM / VRAM) and then re-read by the next op.
    For a Conv2d → BN → ReLU chain this means {c('2 extra round-trips', YELLOW)}
    per chain through the memory bus.

    By fusing {c(n, CYAN, BOLD)} chain(s) we eliminated:
      • {c(f'{rw_eliminated} intermediate memory read/writes', GREEN, BOLD)} to GPU RAM
      • Memory bandwidth freed ≈ {c(f'{rw_eliminated} × feature-map-size', GREEN)}
        (e.g. for a 64-ch 112×112 map: ~{64*112*112*4/1e6:.1f} MB saved per chain
         × {n} chains = ~{n*64*112*112*4/1e6:.1f} MB of unnecessary traffic eliminated)

    Additionally, keeping activations in CUDA registers / L2 cache (rather
    than spilling to HBM) means:
      • Lower {c('latency', YELLOW)} — no wait for DRAM reads
      • Higher {c('arithmetic intensity', YELLOW)} — more FLOPs per byte fetched
      • Better {c('occupancy', YELLOW)} — fewer memory-bound stalls per SM

    {c('Net result:', BOLD)} significant throughput improvement on memory-bandwidth-
    limited workloads (ResNets, MobileNets, EfficientNets).
    """)

    # ── 5. Output ────────────────────────────────────────────────────────────
    print(_section("Output"))
    print(f"    {c('✔', GREEN, BOLD)}  Optimized graph written to:")
    print(f"       {c(os.path.abspath(out_path), CYAN)}\n")
    print(c(f"  {'─' * (_BANNER_WIDTH - 2)}", DIM))
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch DAG Optimizer Agent")
    parser.add_argument("--input",  default="output/graph.json",
                        help="Path to the source graph JSON")
    parser.add_argument("--output", default="output/optimized_graph.json",
                        help="Where to write the optimized graph JSON")
    args = parser.parse_args()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    with open(args.input, encoding="utf-8") as fh:
        raw = json.load(fh)

    gs = GraphState(raw)
    original_nodes = len(gs.nodes)
    original_edges = sum(len(v) for v in gs.fwd.values())

    # ── 2. Pattern matching ───────────────────────────────────────────────────
    chains = find_conv_bn_relu_chains(gs)

    # ── 3. Fuse ─────────────────────────────────────────────────────────────
    fused_ids: list[str] = []
    for conv_id, bn_id, relu_id in chains:
        fid = fuse_chain(gs, conv_id, bn_id, relu_id)
        fused_ids.append(fid)

    optimized_nodes = len(gs.nodes)
    optimized_edges = sum(len(v) for v in gs.fwd.values())

    # ── 4. Export ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    payload = gs.to_json()
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # ── 5. Report ────────────────────────────────────────────────────────────
    print_report(
        original_nodes  = original_nodes,
        original_edges  = original_edges,
        chains          = chains,
        fused_ids       = fused_ids,
        optimized_nodes = optimized_nodes,
        optimized_edges = optimized_edges,
        out_path        = args.output,
    )


if __name__ == "__main__":
    main()
