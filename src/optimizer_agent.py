"""
optimizer_agent.py
------------------
Phase 4 version: Multi-pass optimisation pipeline using PassManager.

Pipeline (in order):
  1. ConvBNReLUFusionPass   — fuse Conv+BN+ReLU chains (Phase 2 logic)
  2. DeadNodeEliminationPass — BFS reachability, remove dead nodes
  3. ConvBNFoldingPass       — fold remaining Conv+BN pairs
  4. MemoryFootprintPass     — annotate nodes with activation memory

Two output files:
  output/optimized_graph.json        (after pass 1 — Conv-BN-ReLU fusion only)
  output/final_optimized_graph.json  (after all 4 passes)

Run:
  python src/optimizer_agent.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from typing import Any

from src.optimizer_passes import (
    ConvBNReLUFusionPass,
    DeadNodeEliminationPass,
    ConvBNFoldingPass,
    MemoryFootprintPass,
    PassManager,
    PassResult,
)

# ── ANSI colours ─────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
DIM     = "\033[2m"

def c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + RESET


# ─────────────────────────────────────────────────────────────────────────────
# In-memory graph (identical to the version in original optimizer_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

class GraphState:
    def __init__(self, raw: dict) -> None:
        self.metadata: dict[str, Any] = deepcopy(raw["metadata"])
        self.nodes: dict[str, dict] = {n["node_id"]: deepcopy(n) for n in raw["nodes"]}
        self.topo: list[str]        = [n["node_id"] for n in raw["nodes"]]
        self.fwd:  dict[str, list[str]] = {nid: [] for nid in self.nodes}
        self.rev:  dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for edge in raw["edges"]:
            s, d = edge["src"], edge["dst"]
            if d not in self.fwd[s]: self.fwd[s].append(d)
            if s not in self.rev[d]: self.rev[d].append(s)

    def successors(self, nid: str) -> list[str]:
        return self.fwd.get(nid, [])

    def predecessors(self, nid: str) -> list[str]:
        return self.rev.get(nid, [])

    def module_class(self, nid: str) -> str:
        return self.nodes[nid].get("kwargs", {}).get("module_class", "")

    def _remove_node(self, nid: str) -> None:
        for succ in self.fwd.pop(nid, []):
            self.rev[succ] = [p for p in self.rev[succ] if p != nid]
        for pred in self.rev.pop(nid, []):
            self.fwd[pred] = [s for s in self.fwd[pred] if s != nid]
        self.nodes.pop(nid, None)

    def _add_node(self, node_dict: dict) -> None:
        nid = node_dict["node_id"]
        self.nodes[nid] = node_dict
        self.fwd.setdefault(nid, [])
        self.rev.setdefault(nid, [])

    def _add_edge(self, src: str, dst: str) -> None:
        if dst not in self.fwd[src]: self.fwd[src].append(dst)
        if src not in self.rev[dst]: self.rev[dst].append(src)

    def to_json(self, label: str = "") -> dict:
        live_set  = set(self.nodes)
        topo_live = [nid for nid in self.topo if nid in live_set]
        fused     = [nid for nid in self.nodes if nid not in set(topo_live)]
        topo_live.extend(fused)
        num_edges = sum(len(v) for v in self.fwd.values())
        return {
            "metadata": {
                **self.metadata,
                "graph_name":   self.metadata["graph_name"] + (f" ({label})" if label else ""),
                "num_nodes":    len(self.nodes),
                "num_edges":    num_edges,
                "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "nodes": [self.nodes[nid] for nid in topo_live],
            "edges": [{"src": s, "dst": d}
                      for s, dsts in self.fwd.items() for d in dsts],
        }

    def clone_raw(self) -> dict:
        """Return a JSON-serialisable dict copy of the current state."""
        return json.loads(json.dumps(self.to_json()))


# ─────────────────────────────────────────────────────────────────────────────
# Export helper
# ─────────────────────────────────────────────────────────────────────────────

def export(gs: GraphState, path: str, label: str = "") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = gs.to_json(label)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal report
# ─────────────────────────────────────────────────────────────────────────────

_W = 66

def _banner(title: str) -> str:
    pad   = _W - len(title) - 4
    left  = pad // 2
    right = pad - left
    return c(f"╔{'═'*(_W-2)}╗\n║  {' '*left}{title}{' '*right}  ║\n╚{'═'*(_W-2)}╝", CYAN, BOLD)

def _section(title: str) -> str:
    return c(f"\n  ── {title} {'─'*max(0,52-len(title))}", BLUE, BOLD)


def print_report(original_n: int, original_e: int,
                 results: list[PassResult], gs: GraphState,
                 out_path: str) -> None:

    final_n = len(gs.nodes)
    final_e = sum(len(v) for v in gs.fwd.values())

    print()
    print(_banner("PyTorch DAG Optimizer  ·  Phase 4 Pipeline Report"))
    print()

    # ── Original stats ────────────────────────────────────────────────────────
    print(_section("Original Graph"))
    print(f"    Nodes : {c(original_n, YELLOW, BOLD)}   Edges : {c(original_e, YELLOW, BOLD)}")

    # ── Per-pass breakdown ────────────────────────────────────────────────────
    print(_section("Pass Pipeline"))
    for i, r in enumerate(results, 1):
        delta_n = r.nodes_before - r.nodes_after
        delta_e = r.edges_before - r.edges_after
        status  = c("✔", GREEN, BOLD) if delta_n >= 0 else c("✗", RED)
        removed = (c(f"-{delta_n} nodes, -{delta_e} edges", RED)
                   if delta_n > 0 else c("no graph change", DIM))
        print(f"\n    {c(f'Pass {i}', BOLD)}: {c(r.pass_name, MAGENTA)}")
        print(f"      {status}  {removed}")
        print(f"         {r.nodes_before} → {r.nodes_after} nodes  |  "
              f"{r.edges_before} → {r.edges_after} edges")
        for note in r.notes[:6]:
            print(f"         {c('·', DIM)} {note}")
        if len(r.notes) > 6:
            print(f"         {c(f'  … and {len(r.notes)-6} more', DIM)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(_section("Overall Graph Reduction"))
    total_nodes_removed = original_n - final_n
    total_edges_removed = original_e - final_e
    pct_n = total_nodes_removed / original_n * 100
    pct_e = total_edges_removed / original_e * 100
    print(f"    Nodes : {c(original_n, YELLOW)} → {c(final_n, GREEN, BOLD)}  "
          f"({c(f'-{total_nodes_removed} / {pct_n:.1f}%', GREEN)})")
    print(f"    Edges : {c(original_e, YELLOW)} → {c(final_e, GREEN, BOLD)}  "
          f"({c(f'-{total_edges_removed} / {pct_e:.1f}%', GREEN)})")

    # ── Memory analysis ───────────────────────────────────────────────────────
    mem = gs.metadata.get("memory_analysis")
    if mem:
        print(_section("Memory Footprint Analysis"))
        peak_val = mem["peak_live_mb"]
        print(f"    Peak live memory : {c(f'{peak_val} MB', CYAN, BOLD)}")
        print(f"    Top activation tensors:")
        for entry in mem["top_consumers"]:
            mem_val = entry["memory_mb"]
            print(f"      {c('▸', DIM)} {entry['node_id']:40s} {c(f'{mem_val} MB', YELLOW)}")

    # ── Output ────────────────────────────────────────────────────────────────
    print(_section("Output"))
    print(f"    {c('✔', GREEN, BOLD)}  Final graph → {c(os.path.abspath(out_path), CYAN)}")
    print(c(f"\n  {'─'*(_W-2)}", DIM))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Multi-Pass DAG Optimizer")
    parser.add_argument("--input",     default="output/graph.json")
    parser.add_argument("--output",    default="output/optimized_graph.json",
                        help="Path for Phase-2-compatible single-pass output")
    parser.add_argument("--final-out", default="output/final_optimized_graph.json",
                        help="Path for final multi-pass output")
    args = parser.parse_args()

    # ── Load original graph ───────────────────────────────────────────────────
    with open(args.input, encoding="utf-8") as fh:
        raw = json.load(fh)

    original_n = raw["metadata"]["num_nodes"]
    original_e = raw["metadata"]["num_edges"]

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = PassManager([
        ConvBNReLUFusionPass(),
        DeadNodeEliminationPass(),
        ConvBNFoldingPass(),
        MemoryFootprintPass(),
    ])

    # ── Run all passes ────────────────────────────────────────────────────────
    gs = GraphState(raw)
    results = pipeline.run(gs)

    # Export intermediate (after pass 1 only) for dashboard backwards compat
    gs1 = GraphState(raw)
    ConvBNReLUFusionPass().run(gs1)
    export(gs1, args.output, "Optimized")

    # Export final (all passes)
    export(gs, args.final_out, "Final Optimized")

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(original_n, original_e, results, gs, args.final_out)


if __name__ == "__main__":
    main()
