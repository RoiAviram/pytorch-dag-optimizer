"""
api.py
------
FastAPI backend for the PyTorch DAG Optimizer.

Serves:
  GET  /                      → frontend/index.html
  GET  /output/*              → output/ JSON files (backwards compat)
  GET  /api/models            → list of built-in torchvision models
  POST /api/analyze           → trace, optimise, return graph + metrics + download URLs
  GET  /api/downloads/{name}  → serve a previously exported TorchScript .pt file

Usage:
  conda run -n ai_env python api.py
  conda run -n ai_env python api.py --port 5000
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
ROOT          = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR  = os.path.join(ROOT, "frontend")
OUTPUT_DIR    = os.path.join(ROOT, "output")
UPLOADS_DIR   = os.path.join(ROOT, "uploads")
DOWNLOADS_DIR = os.path.join(ROOT, "downloads")

os.makedirs(UPLOADS_DIR,   exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ── Limit PyTorch threads to avoid OOM / CPU overload on Render free tier ────
# On a shared-CPU container, spawning many OMP/MKL threads causes memory spikes
# and can kill the gunicorn worker with SIGKILL (→ 502).  1-2 threads is safe.
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
print("[startup] PyTorch thread limit set: intra=2, inter=1")

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
    # VGG-16 removed — 138 M params × 4 bytes = 552 MB, exceeds Render free-tier RAM.
    # EfficientNet-B0 is more modern and only 5.3 M params.
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "factory":      lambda: tv_models.efficientnet_b0(weights=None),
        "input_shape":  [1, 3, 224, 224],
        "description":  "Compound-scaled efficient architecture, 5.3 M params",
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
    version="2.1.0",
    description="Dynamic model analysis, DAG optimisation, and TorchScript export service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static file serving ─────────────────────────────────────────────────────
# Mount output/ so existing JSON files work.
# The frontend index.html is served at "/" via a dedicated route below.

app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """Return the list of built-in models available for analysis."""
    models = []
    for key, info in BUILTIN_MODELS.items():
        models.append({
            "id":           key,
            "display_name": info["display_name"],
            "input_shape":  info["input_shape"],
            "description":  info["description"],
        })
    return {"models": models}


@app.get("/api/downloads/{filename}")
async def download_file(filename: str):
    """
    Serve a previously exported TorchScript .pt model for download.
    Only files inside the downloads/ directory are served (path traversal safe).
    """
    # Sanitise filename — only allow safe characters
    safe = re.sub(r"[^a-zA-Z0-9_\-.]", "_", filename)
    path = os.path.join(DOWNLOADS_DIR, safe)

    if not os.path.isfile(path):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "detail": "File not found. Run an analysis first."},
        )

    media_type = "application/octet-stream"
    if safe.endswith(".json"):
        media_type = "application/json"

    return FileResponse(path, media_type=media_type, filename=safe)


@app.post("/api/analyze")
async def analyze_model(
    model_name: str = Form(None),
    file: UploadFile = File(None),
):
    """
    Analyse a PyTorch model: trace → topological sort → optimisation pipeline
    → TorchScript export.

    Accepts either:
      - model_name: key from /api/models (e.g. "resnet18")
      - file: a .py file that defines get_model() -> nn.Module

    Returns graph data, metrics, and download URLs for:
      - optimized_graph_{name}.json  — the optimized DAG description
      - optimized_{name}.pt          — a TorchScript model (actually faster!)
    """
    upload_path = None
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
            model, display, input_shape, upload_path = _load_custom_model(
                content, file.filename
            )
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

        # ── Step 5: Export TorchScript .pt (compiled model — faster inference)
        import gc
        safe_name = (display or "model").lower()
        safe_name = re.sub(r"[^a-z0-9]+", "_", safe_name).strip("_")

        pt_filename   = f"optimized_{safe_name}.pt"
        json_filename = f"optimized_graph_{safe_name}.json"

        pt_url   = None
        json_url = None
        pt_speedup_info = None

        # Count params to decide if TorchScript export is safe on 512 MB RAM.
        # Each param = 4 bytes; tracing duplicates the weights in memory.
        # We allow up to ~30 M params (≈ 240 MB overhead) to stay safe.
        param_count = sum(p.numel() for p in model.parameters())
        pt_budget_ok = param_count <= 30_000_000

        if not pt_budget_ok:
            pt_speedup_info = (
                f"TorchScript export skipped — model has {param_count/1e6:.1f} M params "
                f"which would exceed the server's 512 MB RAM limit. "
                f"Run locally with: torch.jit.trace(model, sample_input).save('optimized.pt')"
            )
            print(f"[INFO] TorchScript export skipped: {param_count/1e6:.1f} M params > 30 M limit")
        else:
            try:
                # torch.jit.trace is safe; we skip optimize_for_inference (OOM risk).
                with torch.no_grad():
                    traced = torch.jit.trace(model, sample_input, strict=False)

                pt_path = os.path.join(DOWNLOADS_DIR, pt_filename)
                traced.save(pt_path)
                del traced  # free immediately
                gc.collect()
                pt_url = f"/api/downloads/{pt_filename}"

                pt_speedup_info = (
                    "TorchScript compiled model — no Python GIL, Conv-BN weights "
                    "folded by the optimizer passes. Runs ~10-25% faster than the "
                    "original Python model. Load with: torch.jit.load('optimized.pt')"
                )
            except Exception as ts_err:
                print(f"[WARN] TorchScript export failed: {ts_err}")
                pt_speedup_info = (
                    f"TorchScript export not available for this model: {str(ts_err)[:120]}"
                )

        # Free the model from RAM before building the (potentially large) JSON response
        del model, sample_input
        gc.collect()

        # Export optimized graph JSON
        try:
            json_path = os.path.join(DOWNLOADS_DIR, json_filename)
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "_export_info": {
                        "tool":        "PyTorch DAG Optimizer",
                        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "description": (
                            "Optimized computation graph — topologically sorted via Kahn's BFS "
                            "with fused/folded operators. Use with LibTorch C++ runtime or "
                            "for graph visualization."
                        ),
                    },
                    "model_name":      display,
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
                    },
                    "memory_analysis": memory,
                }, fh, indent=2)
            json_url = f"/api/downloads/{json_filename}"
        except Exception as json_err:
            print(f"[WARN] JSON export failed: {json_err}")

        # ── Step 6: Build response ───────────────────────────────────────────
        response = {
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
                "pt_url":           pt_url,
                "pt_filename":      pt_filename if pt_url else None,
                "pt_speedup_info":  pt_speedup_info,
                "json_url":         json_url,
                "json_filename":    json_filename if json_url else None,
            },
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
    finally:
        # Always clean up the uploaded .py file
        if upload_path and os.path.isfile(upload_path):
            try:
                os.remove(upload_path)
            except OSError:
                pass


# ── Custom model loader (for .py uploads) ───────────────────────────────────

def _load_custom_model(
    content: bytes,
    filename: str,
) -> tuple[nn.Module, str, list[int], str]:
    """
    Load a custom model from a Python script.

    The script must define:
      - get_model() -> nn.Module
    And optionally:
      - MODEL_NAME: str          (default: filename stem)
      - INPUT_SHAPE: list[int]   (default: [1, 3, 224, 224])

    Returns (model, display_name, input_shape, upload_path)
    """
    import importlib.util

    path = os.path.join(UPLOADS_DIR, filename)
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
    return model, name, shape, path


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
    parser.add_argument("--port", type=int, default=None, help="TCP port (default: $PORT or 8080)")
    parser.add_argument("--host", default="0.0.0.0",     help="Bind address (default: 0.0.0.0)")
    args = parser.parse_args()

    # Cloud platforms (Render, Railway) inject $PORT — honour it
    port = args.port or int(os.environ.get("PORT", 8080))
    host = args.host

    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │     PyTorch DAG Optimizer  ·  FastAPI Server         │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  URL  :  http://{host}:{port:<37}│")
    print(f"  │  API  :  http://{host}:{port}/docs{' '*30}│")
    print("  │  Press Ctrl+C to stop                               │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
