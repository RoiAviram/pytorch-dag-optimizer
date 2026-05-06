"""
api.py
------
FastAPI backend for the PyTorch DAG Optimizer.

Serves:
  GET  /                → frontend/index.html
  GET  /output/*        → output/ JSON files (backwards compat)
  GET  /api/models      → list of built-in torchvision models
  POST /api/analyze     → trace, optimise, and return graph + metrics

Usage:
  conda run -n ai_env python api.py
  conda run -n ai_env python api.py --port 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
OUTPUT_DIR   = os.path.join(ROOT, "output")

sys.path.insert(0, ROOT)
from src.dag_core import ComputationDAG, DAGNode          # noqa: E402
from src.graph_extractor import extract_dag                # noqa: E402
from src.optimizer_agent import GraphState, export         # noqa: E402
from src.optimizer_passes import (                         # noqa: E402
    ConvBNReLUFusionPass,
    DeadNodeEliminationPass,
    ConvBNFoldingPass,
    MemoryFootprintPass,
    PassManager,
)

# ── Built-in model registry ─────────────────────────────────────────────────

BUILTIN_MODELS: dict[str, dict[str, Any]] = {
    "resnet18": {
        "display_name": "ResNet-18",
        "factory":      lambda: tv_models.resnet18(weights=None),
        "input_shape":  [1, 3, 224, 224],
        "description":  "18-layer residual network, 11.7 M params",
    },
    "resnet34": {
        "display_name": "ResNet-34",
        "factory":      lambda: tv_models.resnet34(weights=None),
        "input_shape":  [1, 3, 224, 224],
        "description":  "34-layer residual network, 21.8 M params",
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "factory":      lambda: tv_models.resnet50(weights=None),
        "input_shape":  [1, 3, 224, 224],
        "description":  "50-layer residual network (bottleneck blocks), 25.6 M params",
    },
    "vgg16": {
        "display_name": "VGG-16",
        "factory":      lambda: tv_models.vgg16(weights=None),
        "input_shape":  [1, 3, 224, 224],
        "description":  "16-layer VGG, 138 M params",
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

# ── Bandwidth math constants (same as original app.js) ───────────────────────

BYTES_PER_INTERMEDIATE = 64 * 112 * 112 * 4   # ~3.2 MB (stem feature-map)
THROUGHPUT_GAIN_PER_CHAIN_PCT = 1.8            # empirical, conservative


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="PyTorch DAG Optimizer API",
    version="2.0.0",
    description="Dynamic model analysis and DAG optimisation service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static file serving ─────────────────────────────────────────────────────
# Mount output/ and frontend/ so existing JSON files and static assets work.
# The frontend index.html is served at "/" via a dedicated route below.

app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """Return the list of built-in models available for analysis."""
    models = []
    for key, info in BUILTIN_MODELS.items():
        models.append({
            "id":          key,
            "display_name": info["display_name"],
            "input_shape":  info["input_shape"],
            "description":  info["description"],
        })
    return {"models": models}


@app.post("/api/analyze")
async def analyze_model(
    model_name: str = Form(None),
    file: UploadFile = File(None),
):
    """
    Analyse a PyTorch model: trace → topological sort → optimisation pipeline.

    Accepts either:
      - model_name: key from /api/models (e.g. "resnet18")
      - file: a .py file that defines get_model() -> nn.Module
    """
    try:
        # ── Step 1: Resolve the model ────────────────────────────────────────
        if model_name and model_name in BUILTIN_MODELS:
            info        = BUILTIN_MODELS[model_name]
            display     = info["display_name"]
            model       = info["factory"]()
            input_shape = info["input_shape"]
        elif file is not None:
            # Security: only accept .py files, max 1 MB
            if not file.filename.endswith(".py"):
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "detail": "Only .py files are accepted."},
                )
            content = await file.read()
            if len(content) > 1_000_000:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "detail": "File too large (max 1 MB)."},
                )
            # Dynamic import: write to temp, import, call get_model()
            model, display, input_shape = _load_custom_model(content, file.filename)
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "detail": "Provide model_name (built-in) or upload a .py file.",
                },
            )

        model.eval()
        sample_input = torch.randn(*input_shape)

        # ── Step 2: Extract original DAG ─────────────────────────────────────
        dag = extract_dag(model, sample_input, graph_name=display)
        original_raw = dag.to_json()

        original_n = original_raw["metadata"]["num_nodes"]
        original_e = original_raw["metadata"]["num_edges"]

        # ── Step 3: Run optimisation pipeline ────────────────────────────────
        pipeline = PassManager([
            ConvBNReLUFusionPass(),
            DeadNodeEliminationPass(),
            ConvBNFoldingPass(),
            MemoryFootprintPass(),
        ])

        gs = GraphState(original_raw)
        results = pipeline.run(gs)

        optimized_raw = gs.to_json("Optimized")

        final_n = optimized_raw["metadata"]["num_nodes"]
        final_e = optimized_raw["metadata"]["num_edges"]

        # ── Step 4: Compute metrics ──────────────────────────────────────────
        fused_count  = sum(1 for n in optimized_raw["nodes"]
                          if n.get("op_type") == "fused_op")
        folded_count = sum(1 for n in optimized_raw["nodes"]
                          if n.get("op_type") == "folded_op")

        rw_eliminated   = fused_count * 2
        bw_freed_bytes  = fused_count * 2 * BYTES_PER_INTERMEDIATE
        bw_cut_pct      = (rw_eliminated / max(original_e * 2, 1)) * 100
        throughput_gain = fused_count * THROUGHPUT_GAIN_PER_CHAIN_PCT

        node_red_pct = ((original_n - final_n) / max(original_n, 1)) * 100
        edge_red_pct = ((original_e - final_e) / max(original_e, 1)) * 100

        # Human-readable bandwidth
        if bw_freed_bytes >= 1e9:
            bw_human = f"~{bw_freed_bytes / 1e9:.1f} GB"
        elif bw_freed_bytes >= 1e6:
            bw_human = f"~{bw_freed_bytes / 1e6:.1f} MB"
        elif bw_freed_bytes >= 1e3:
            bw_human = f"~{bw_freed_bytes / 1e3:.1f} KB"
        else:
            bw_human = f"{bw_freed_bytes} B"

        # Pass results for the pipeline panel
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

        # Memory analysis (populated by MemoryFootprintPass)
        memory = gs.metadata.get("memory_analysis", {})

        # ── Step 5: Build response ───────────────────────────────────────────
        response = {
            "status":     "success",
            "model_name": display,
            "original_graph":  original_raw,
            "optimized_graph": optimized_raw,
            "metrics": {
                "original_nodes":          original_n,
                "original_edges":          original_e,
                "optimized_nodes":         final_n,
                "optimized_edges":         final_e,
                "node_reduction_pct":      round(node_red_pct, 1),
                "edge_reduction_pct":      round(edge_red_pct, 1),
                "fused_ops":               fused_count,
                "folded_ops":              folded_count,
                "hbm_rw_eliminated":       rw_eliminated,
                "hbm_bandwidth_freed_bytes": bw_freed_bytes,
                "hbm_bandwidth_freed_human": bw_human,
                "hbm_traffic_cut_pct":     round(bw_cut_pct, 1),
                "est_throughput_gain_pct":  round(throughput_gain, 1),
                "pass_results":            pass_results,
            },
            "memory_analysis": memory,
        }

        return JSONResponse(content=response)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": f"Analysis failed: {str(e)}",
            },
        )


# ── Custom model loader (for .py uploads) ───────────────────────────────────

def _load_custom_model(
    content: bytes,
    filename: str,
) -> tuple[nn.Module, str, list[int]]:
    """
    Load a custom model from a Python script.

    The script must define:
      - get_model() -> nn.Module
    And optionally:
      - MODEL_NAME: str          (default: filename stem)
      - INPUT_SHAPE: list[int]   (default: [1, 3, 224, 224])
    """
    import importlib.util
    import tempfile

    uploads_dir = os.path.join(ROOT, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    path = os.path.join(uploads_dir, filename)
    with open(path, "wb") as fh:
        fh.write(content)

    spec = importlib.util.spec_from_file_location("_user_model", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "get_model"):
        raise ValueError(
            "Uploaded .py file must define a `get_model()` function "
            "that returns an nn.Module."
        )

    model = mod.get_model()
    name  = getattr(mod, "MODEL_NAME",  os.path.splitext(filename)[0])
    shape = getattr(mod, "INPUT_SHAPE", [1, 3, 224, 224])
    return model, name, shape


# ── Serve frontend index.html ────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# Mount frontend static files AFTER the explicit routes so they don't
# shadow /api/* or /.  The "html=False" prevents it from serving index.html
# at "/" (we handle that above).
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="PyTorch DAG Optimizer API Server")
    parser.add_argument("--port", type=int, default=8080, help="TCP port (default 8080)")
    parser.add_argument("--host", default="127.0.0.1",    help="Bind address")
    args = parser.parse_args()

    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │     PyTorch DAG Optimizer  ·  FastAPI Server         │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  URL  :  http://{args.host}:{args.port:<37}│")
    print(f"  │  API  :  http://{args.host}:{args.port}/docs{' '*30}│")
    print("  │  Press Ctrl+C to stop                               │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
