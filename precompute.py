#!/usr/bin/env python3
"""
precompute.py
-------------
Run this script LOCALLY (not on Render) to pre-compute analysis results for
all built-in models.  The results are saved as JSON files in precomputed/
and committed to the repo.  The server loads these at startup with zero
model loading → zero OOM risk.

Usage:
    conda run -n ai_env python precompute.py
"""

import gc
import json
import os
import re
import sys
import time

import torch
import torchvision.models as tv_models

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.graph_extractor import extract_dag
from src.optimizer_agent import GraphState
from src.optimizer_passes import (
    ConvBNReLUFusionPass,
    DeadNodeEliminationPass,
    ConvBNFoldingPass,
    MemoryFootprintPass,
    PassManager,
)

# ── Models to pre-compute (3 diverse architectures) ──────────────────────────
MODELS = {
    "resnet18": {
        "display_name": "ResNet-18",
        "factory":      lambda: tv_models.resnet18(weights=None),
        "input_shape":  [1, 3, 224, 224],
        "description":  "18-layer residual network, 11.7 M params",
    },
    "mobilenet_v2": {
        "display_name": "MobileNet V2",
        "factory":      lambda: tv_models.mobilenet_v2(weights=None),
        "input_shape":  [1, 3, 224, 224],
        "description":  "Inverted-residual mobile architecture, 3.4 M params",
    },
    "squeezenet1_0": {
        "display_name": "SqueezeNet 1.0",
        "factory":      lambda: tv_models.squeezenet1_0(weights=None),
        "input_shape":  [1, 3, 224, 224],
        "description":  "Compact fire-module architecture, 1.2 M params",
    },
}

BYTES_PER_INTERMEDIATE        = 64 * 112 * 112 * 4
THROUGHPUT_GAIN_PER_CHAIN_PCT = 1.8

OUT_DIR = os.path.join(ROOT, "precomputed")
os.makedirs(OUT_DIR, exist_ok=True)


def analyze_one(model_id: str, info: dict) -> dict:
    """Full analysis pipeline for one model. Returns the API response dict."""
    display     = info["display_name"]
    input_shape = info["input_shape"]
    model       = info["factory"]()
    model.eval()

    # ShapeProp input: 32×32 spatial (saves activation memory)
    sp_shape = list(input_shape)
    if len(sp_shape) >= 4:
        sp_shape[-2] = min(sp_shape[-2], 32)
        sp_shape[-1] = min(sp_shape[-1], 32)
    sp_input = torch.zeros(*sp_shape)

    # Step 1: Extract DAG
    with torch.no_grad():
        dag = extract_dag(model, sp_input, graph_name=display)
    original_raw = dag.to_json()
    del dag, sp_input
    gc.collect()

    original_n = original_raw["metadata"]["num_nodes"]
    original_e = original_raw["metadata"]["num_edges"]

    # Step 2: Optimisation pipeline
    pipeline = PassManager([
        ConvBNReLUFusionPass(),
        DeadNodeEliminationPass(),
        ConvBNFoldingPass(),
        MemoryFootprintPass(),
    ])
    gs      = GraphState(original_raw)
    results = pipeline.run(gs)

    optimized_raw = gs.to_json("Optimized")
    final_n = optimized_raw["metadata"]["num_nodes"]
    final_e = optimized_raw["metadata"]["num_edges"]

    # Step 3: Metrics
    fused_count  = sum(1 for n in optimized_raw["nodes"] if n.get("op_type") == "fused_op")
    folded_count = sum(1 for n in optimized_raw["nodes"] if n.get("op_type") == "folded_op")

    rw_eliminated   = fused_count * 2
    bw_freed_bytes  = fused_count * 2 * BYTES_PER_INTERMEDIATE
    bw_cut_pct      = (rw_eliminated / max(original_e * 2, 1)) * 100
    throughput_gain = fused_count * THROUGHPUT_GAIN_PER_CHAIN_PCT
    node_red_pct    = ((original_n - final_n) / max(original_n, 1)) * 100
    edge_red_pct    = ((original_e - final_e) / max(original_e, 1)) * 100

    if bw_freed_bytes >= 1e6:
        bw_human = f"~{bw_freed_bytes / 1e6:.1f} MB"
    elif bw_freed_bytes >= 1e3:
        bw_human = f"~{bw_freed_bytes / 1e3:.1f} KB"
    else:
        bw_human = f"{bw_freed_bytes} B"

    pass_results = []
    for r in results:
        pass_results.append({
            "pass_name":     r.pass_name,
            "nodes_before":  r.nodes_before,
            "nodes_after":   r.nodes_after,
            "edges_before":  r.edges_before,
            "edges_after":   r.edges_after,
            "nodes_removed": r.nodes_removed,
            "edges_removed": r.edges_removed,
            "notes":         r.notes[:10],
        })

    memory   = gs.metadata.get("memory_analysis", {})
    safe_name = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")

    # Clean up model
    del model
    gc.collect()

    return {
        "status":     "success",
        "model_name": display,
        "original_graph":  original_raw,
        "optimized_graph": optimized_raw,
        "metrics": {
            "original_nodes":            original_n,
            "original_edges":            original_e,
            "optimized_nodes":           final_n,
            "optimized_edges":           final_e,
            "node_reduction_pct":        round(node_red_pct, 1),
            "edge_reduction_pct":        round(edge_red_pct, 1),
            "fused_ops":                 fused_count,
            "folded_ops":                folded_count,
            "hbm_rw_eliminated":         rw_eliminated,
            "hbm_bandwidth_freed_bytes": bw_freed_bytes,
            "hbm_bandwidth_freed_human": bw_human,
            "hbm_traffic_cut_pct":       round(bw_cut_pct, 1),
            "est_throughput_gain_pct":   round(throughput_gain, 1),
            "pass_results":              pass_results,
        },
        "memory_analysis": memory,
        "downloads": {
            "pt_url":          None,
            "pt_filename":     None,
            "pt_speedup_info": (
                f"TorchScript export available when running locally. "
                f"Use: model = {display.replace(' ', '')}(); "
                f"torch.jit.trace(model, torch.randn({input_shape})).save('optimized.pt')"
            ),
            "json_url":        f"/api/downloads/optimized_graph_{safe_name}.json",
            "json_filename":   f"optimized_graph_{safe_name}.json",
        },
    }


def main():
    print("═" * 60)
    print("  Pre-computing built-in model analyses")
    print("═" * 60)

    for model_id, info in MODELS.items():
        print(f"\n[{model_id}] Analyzing {info['display_name']} …")
        t0 = time.time()

        result = analyze_one(model_id, info)

        # Save the full API response JSON
        out_path = os.path.join(OUT_DIR, f"{model_id}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

        # Also save the optimized graph JSON for download
        dl_path = os.path.join(ROOT, "downloads",
                               result["downloads"]["json_filename"])
        os.makedirs(os.path.dirname(dl_path), exist_ok=True)
        with open(dl_path, "w", encoding="utf-8") as fh:
            json.dump({
                "_export_info": {
                    "tool":        "PyTorch DAG Optimizer",
                    "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "description": (
                        "Optimized computation graph — topologically sorted via "
                        "Kahn's BFS with fused/folded operators."
                    ),
                },
                "model_name":      result["model_name"],
                "optimized_graph": result["optimized_graph"],
                "metrics":         result["metrics"],
                "memory_analysis": result["memory_analysis"],
            }, fh, indent=2)

        elapsed = time.time() - t0
        m = result["metrics"]
        print(f"  ✓ Nodes: {m['original_nodes']} → {m['optimized_nodes']}  "
              f"({m['node_reduction_pct']}% reduction)")
        print(f"  ✓ Fused: {m['fused_ops']}  Folded: {m['folded_ops']}")
        print(f"  ✓ Saved: {out_path}  ({elapsed:.1f}s)")

    print(f"\n{'═' * 60}")
    print(f"  Done — {len(MODELS)} models pre-computed in precomputed/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
