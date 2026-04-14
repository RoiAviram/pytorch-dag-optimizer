"""
graph_extractor.py
------------------
Uses torch.fx to symbolically trace a ResNet-18 model and populate our
custom ComputationDAG, then exports the result to output/graph.json.

Pipeline:
  1. Load torchvision.models.resnet18() in eval mode
  2. torch.fx.symbolic_trace  →  fx.GraphModule
  3. ShapeProp pass           →  annotate each node with its output tensor shape
  4. Iterate fx nodes         →  create DAGNode + edges in ComputationDAG
  5. dag.export_json(...)     →  write output/graph.json

Run:
  python src/graph_extractor.py
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any

import torch
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

import torchvision.models as tv_models

# Make sure sibling modules are importable when run from repo root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dag_core import ComputationDAG, DAGNode  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: resolve a human-readable target string from an fx.Node
# ---------------------------------------------------------------------------

def _resolve_target(node: torch.fx.Node) -> str:
    """Return a clean string representation of node.target."""
    if callable(node.target):
        return getattr(node.target, "__name__", str(node.target))
    return str(node.target)


# ---------------------------------------------------------------------------
# Helper: extract output shape from ShapeProp metadata
# ---------------------------------------------------------------------------

def _get_shape(node: torch.fx.Node) -> list[int] | None:
    """
    ShapeProp stores a TensorMetadata (or tuple thereof) in node.meta['tensor_meta'].
    We flatten the first tensor's shape for simplicity.
    """
    meta = node.meta.get("tensor_meta", None)
    if meta is None:
        return None
    # May be a TensorMetadata namedtuple or a tuple of them
    if hasattr(meta, "shape"):
        return list(meta.shape)
    if isinstance(meta, (list, tuple)) and len(meta) > 0:
        first = meta[0]
        if hasattr(first, "shape"):
            return list(first.shape)
    return None


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_dag(model: torch.nn.Module, sample_input: torch.Tensor,
                graph_name: str = "ResNet18") -> ComputationDAG:
    """
    Symbolically trace *model* with torch.fx, run ShapeProp, and build a
    ComputationDAG from the resulting IR.

    Args:
        model:        The nn.Module to trace (must be eval-mode for BN stability).
        sample_input: A representative input tensor (used by ShapeProp).
        graph_name:   Name embedded in the DAG metadata.

    Returns:
        A fully populated ComputationDAG.
    """
    print(f"[Extractor] Tracing {graph_name} with torch.fx …")
    gm: torch.fx.GraphModule = torch.fx.symbolic_trace(model)

    # ----- Shape propagation -----
    print("[Extractor] Running ShapeProp to annotate output shapes …")
    sp = ShapeProp(gm)
    sp.propagate(sample_input)

    # ----- Build DAG -----
    dag = ComputationDAG(name=graph_name)

    # Pass 1: register all nodes
    for fx_node in gm.graph.nodes:
        target_str = _resolve_target(fx_node)
        shape      = _get_shape(fx_node)

        # Collect auxiliary metadata
        extra: dict[str, Any] = {}
        if fx_node.op == "call_module":
            # Record the actual nn.Module class for richer analysis
            submodule = dict(gm.named_modules()).get(str(fx_node.target))
            if submodule is not None:
                extra["module_class"] = type(submodule).__name__
        if fx_node.kwargs:
            # Capture any kwargs that are JSON-serialisable primitives
            for k, v in fx_node.kwargs.items():
                if isinstance(v, (int, float, str, bool, list, type(None))):
                    extra[k] = v

        dag_node = DAGNode(
            node_id      = fx_node.name,
            op_type      = fx_node.op,
            name         = target_str,
            output_shape = shape,
            kwargs       = extra,
        )
        dag.add_node(dag_node)

    # Pass 2: add edges (data-flow from each arg that is itself an fx.Node)
    for fx_node in gm.graph.nodes:
        for arg in fx_node.args:
            if isinstance(arg, torch.fx.Node):
                dag.add_edge(arg.name, fx_node.name)
        # Some ops pass node-valued kwargs too (e.g., some concat-like ops)
        for v in fx_node.kwargs.values():
            if isinstance(v, torch.fx.Node):
                dag.add_edge(v.name, fx_node.name)

    print(f"[Extractor] DAG built: {dag}")
    return dag


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "output", "graph.json"
    )

    # ------ Model setup ------
    model = tv_models.resnet18(weights=None)
    model.eval()

    # Representative ImageNet-like input: batch=1, RGB, 224×224
    sample_input = torch.randn(1, 3, 224, 224)

    # ------ Extract ------
    dag = extract_dag(model, sample_input, graph_name="ResNet18")

    # ------ Export ------
    dag.export_json(output_path)

    # ------ Pretty-print a preview ------
    abs_path = os.path.abspath(output_path)
    with open(abs_path, encoding="utf-8") as fh:
        data = json.load(fh)

    meta = data["metadata"]
    print("\n" + "=" * 60)
    print("  Graph JSON — Quick Preview")
    print("=" * 60)
    print(f"  Graph : {meta['graph_name']}")
    print(f"  Nodes : {meta['num_nodes']}")
    print(f"  Edges : {meta['num_edges']}")
    print(f"  File  : {abs_path}")
    print("-" * 60)
    print("  First 5 nodes (topo order):")
    for n in data["nodes"][:5]:
        shape_str = str(n["output_shape"]) if n["output_shape"] else "N/A"
        print(f"    [{n['op_type']:15s}]  {n['node_id']:30s}  shape={shape_str}")
    print("  …")
    print("  Last 5 nodes:")
    for n in data["nodes"][-5:]:
        shape_str = str(n["output_shape"]) if n["output_shape"] else "N/A"
        print(f"    [{n['op_type']:15s}]  {n['node_id']:30s}  shape={shape_str}")
    print("=" * 60)
    print("  First 5 edges:")
    for e in data["edges"][:5]:
        print(f"    {e['src']}  →  {e['dst']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
