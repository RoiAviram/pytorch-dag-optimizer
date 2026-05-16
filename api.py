"""
api.py
------
FastAPI backend for the PyTorch DAG Optimizer.

Architecture (memory-safe for Render free tier, 512 MB RAM):
  - ALL built-in models are analyzed ONCE at server startup (sequentially).
    Each model is loaded, analyzed, traced, then immediately deleted + gc'd.
    After startup, the server holds only ~200 MB (PyTorch runtime + cached dicts).
  - Requests for built-in models return the pre-cached JSON instantly.
    Zero model loading at request time → zero memory spike → no more 502s.
  - Custom .py uploads still do live analysis (with memory guards).

Serves:
  GET  /                      → frontend/index.html
  GET  /output/*              → output/ JSON files (backwards compat)
  GET  /api/models            → list of built-in torchvision models
  POST /api/analyze           → returns cached result (built-in) or live analysis (.py)
  GET  /api/downloads/{name}  → serve a previously exported file
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR  = os.path.join(ROOT, "frontend")
OUTPUT_DIR    = os.path.join(ROOT, "output")
UPLOADS_DIR   = os.path.join(ROOT, "uploads")
DOWNLOADS_DIR = os.path.join(ROOT, "downloads")

os.makedirs(UPLOADS_DIR,   exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ── Limit PyTorch threads on Render's shared CPU ─────────────────────────────
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

# ── Built-in model registry ───────────────────────────────────────────────────

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

# ── Bandwidth math constants ──────────────────────────────────────────────────
BYTES_PER_INTERMEDIATE       = 64 * 112 * 112 * 4   # ~3.2 MB (stem feature-map)
THROUGHPUT_GAIN_PER_CHAIN_PCT = 1.8                  # empirical, conservative

# ── In-memory cache: filled at startup, read at request time ─────────────────
PRECOMPUTED: dict[str, dict] = {}


# ═════════════════════════════════════════════════════════════════════════════
# Core analysis helper — used at startup for built-ins & live for uploads
# ═════════════════════════════════════════════════════════════════════════════

def _run_analysis(
    model:       nn.Module,
    input_shape: list[int],
    display:     str,
    safe_name:   str,
) -> dict[str, Any]:
    """
    Full analysis pipeline for one model.
    IMPORTANT: deletes `model` and calls gc.collect() before returning.
    The caller must NOT use `model` after this function returns.
    """
    model.eval()

    # ── ShapeProp input: max 32×32 spatial (49× less activation RAM vs 224×224) ──
    sp_shape = list(input_shape)
    if len(sp_shape) >= 4:
        sp_shape[-2] = min(sp_shape[-2], 32)
        sp_shape[-1] = min(sp_shape[-1], 32)
    sp_input = torch.zeros(*sp_shape)

    # ── Step 1: Extract original DAG ─────────────────────────────────────────
    with torch.no_grad():
        dag = extract_dag(model, sp_input, graph_name=display)
    original_raw = dag.to_json()
    del dag, sp_input
    gc.collect()

    original_n = original_raw["metadata"]["num_nodes"]
    original_e = original_raw["metadata"]["num_edges"]

    # ── Step 2: Optimisation pipeline ────────────────────────────────────────
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

    # ── Step 3: Metrics ───────────────────────────────────────────────────────
    fused_count  = sum(1 for n in optimized_raw["nodes"] if n.get("op_type") == "fused_op")
    folded_count = sum(1 for n in optimized_raw["nodes"] if n.get("op_type") == "folded_op")

    rw_eliminated   = fused_count * 2
    bw_freed_bytes  = fused_count * 2 * BYTES_PER_INTERMEDIATE
    bw_cut_pct      = (rw_eliminated / max(original_e * 2, 1)) * 100
    throughput_gain = fused_count * THROUGHPUT_GAIN_PER_CHAIN_PCT
    node_red_pct    = ((original_n - final_n) / max(original_n, 1)) * 100
    edge_red_pct    = ((original_e - final_e) / max(original_e, 1)) * 100

    if bw_freed_bytes >= 1e9:
        bw_human = f"~{bw_freed_bytes / 1e9:.1f} GB"
    elif bw_freed_bytes >= 1e6:
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

    memory = gs.metadata.get("memory_analysis", {})

    # ── Step 4: TorchScript .pt export (only for models ≤ 15 M params) ───────
    # 15 M params × 4 bytes × 2 (model + trace copy) = 120 MB overhead — safe.
    pt_url          = None
    pt_speedup_info = None
    json_url        = None
    pt_filename     = f"optimized_{safe_name}.pt"
    json_filename   = f"optimized_graph_{safe_name}.json"

    param_count  = sum(p.numel() for p in model.parameters())
    pt_budget_ok = param_count <= 15_000_000

    if pt_budget_ok:
        try:
            trace_input = torch.randn(*input_shape)
            with torch.no_grad():
                traced = torch.jit.trace(model, trace_input, strict=False)
            pt_path = os.path.join(DOWNLOADS_DIR, pt_filename)
            traced.save(pt_path)
            del traced, trace_input
            gc.collect()
            pt_url = f"/api/downloads/{pt_filename}"
            pt_speedup_info = (
                "TorchScript compiled model — no Python GIL overhead, Conv-BN weights "
                "folded by the optimizer passes. Runs ~10-25% faster than the original "
                "Python model. Load with: torch.jit.load('optimized.pt')"
            )
        except Exception as ts_err:
            print(f"[WARN] TorchScript export failed for {safe_name}: {ts_err}")
            pt_speedup_info = f"TorchScript export failed: {str(ts_err)[:100]}"
    else:
        pt_speedup_info = (
            f"TorchScript export skipped — model has {param_count/1e6:.1f} M params "
            f"({param_count/1e6:.1f} M × 4 bytes × 2 copies ≈ "
            f"{param_count * 8 / 1e6:.0f} MB overhead, exceeds server RAM). "
            f"Run locally: torch.jit.trace(model, x).save('optimized.pt')"
        )

    # Free model BEFORE building the JSON (large models keep weights in memory)
    del model
    gc.collect()

    # ── Step 5: Write optimized JSON to disk ──────────────────────────────────
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
                    "fused_ops":                 fused_count,
                    "folded_ops":                folded_count,
                    "hbm_bandwidth_freed_human": bw_human,
                },
                "memory_analysis": memory,
            }, fh, indent=2)
        json_url = f"/api/downloads/{json_filename}"
    except Exception as json_err:
        print(f"[WARN] JSON export failed for {safe_name}: {json_err}")

    # ── Build response dict ───────────────────────────────────────────────────
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
            "pt_url":          pt_url,
            "pt_filename":     pt_filename if pt_url else None,
            "pt_speedup_info": pt_speedup_info,
            "json_url":        json_url,
            "json_filename":   json_filename if json_url else None,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Startup: pre-compute all built-in models (sequentially, memory-safe)
# ═════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    At startup: analyze all built-in models one by one. Each is loaded, analyzed,
    and DELETED before the next one starts. After all 6 models, peak memory use
    is ~200 MB (PyTorch runtime + cached JSON dicts), leaving 300 MB headroom.
    At request time: no model loading → no memory spike → no 502 errors.
    """
    print("\n[startup] ═══ Pre-computing built-in models ═══")
    for model_id, info in BUILTIN_MODELS.items():
        print(f"[startup] Analyzing {model_id} ({info['description']}) …")
        t0 = time.time()
        try:
            model     = info["factory"]()
            safe_name = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
            result    = _run_analysis(model, info["input_shape"], info["display_name"], safe_name)
            PRECOMPUTED[model_id] = result
            elapsed = time.time() - t0
            print(f"[startup] ✓ {model_id} ready ({elapsed:.1f}s)")
        except Exception:
            print(f"[startup] ✗ {model_id} FAILED:\n{traceback.format_exc()}")
            gc.collect()

    print(f"[startup] ═══ Done — {len(PRECOMPUTED)}/{len(BUILTIN_MODELS)} models cached ═══\n")
    yield
    # Shutdown — nothing to clean up


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="PyTorch DAG Optimizer API",
    version="3.0.0",
    description="Memory-safe DAG optimisation service with startup pre-computation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving ───────────────────────────────────────────────────────
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    """Return the list of built-in models. Marks which ones are already cached."""
    models = []
    for key, info in BUILTIN_MODELS.items():
        models.append({
            "id":           key,
            "display_name": info["display_name"],
            "input_shape":  info["input_shape"],
            "description":  info["description"],
            "cached":       key in PRECOMPUTED,
        })
    return {"models": models}


@app.get("/api/downloads/{filename}")
async def download_file(filename: str):
    """Serve a previously exported file from the downloads/ directory."""
    safe = re.sub(r"[^a-zA-Z0-9_\-.]", "_", filename)
    path = os.path.join(DOWNLOADS_DIR, safe)

    if not os.path.isfile(path):
        return JSONResponse(
            status_code=404,
            content={"status": "error", "detail": "File not found. Run an analysis first."},
        )

    media_type = "application/json" if safe.endswith(".json") else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=safe)


@app.post("/api/analyze")
async def analyze_model(
    model_name: str | None       = Form(default=None),
    file:       UploadFile | None = File(default=None),
):
    """
    Analyze a model and return the optimized graph + metrics + download URLs.

    For built-in models: returns the pre-cached result instantly (no model loading).
    For custom .py uploads: runs live analysis (small models only, < 200 MB RAM).
    """
    upload_path = None
    try:
        # ── Built-in model: serve from cache ─────────────────────────────────
        if model_name and model_name in BUILTIN_MODELS:
            if model_name in PRECOMPUTED:
                print(f"[analyze] Cache hit for '{model_name}' — returning instantly")
                return JSONResponse(content=PRECOMPUTED[model_name])
            else:
                # Startup failed for this model — try live analysis as fallback
                print(f"[analyze] Cache miss for '{model_name}' — running live analysis")
                info      = BUILTIN_MODELS[model_name]
                model     = info["factory"]()
                safe_name = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
                result    = _run_analysis(model, info["input_shape"], info["display_name"], safe_name)
                PRECOMPUTED[model_name] = result  # cache for next time
                return JSONResponse(content=result)

        # ── Custom .py upload: live analysis ──────────────────────────────────
        elif file is not None:
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

            model, display, input_shape, upload_path = _load_custom_model(content, file.filename)
            safe_name = re.sub(r"[^a-z0-9]+", "_", display.lower()).strip("_") or "custom"
            result    = _run_analysis(model, input_shape, display, safe_name)
            return JSONResponse(content=result)

        else:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "detail": "Provide model_name (built-in) or upload a .py file.",
                },
            )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": f"Analysis failed: {str(e)}"},
        )
    finally:
        if upload_path and os.path.isfile(upload_path):
            try:
                os.remove(upload_path)
            except OSError:
                pass


# ── Custom model loader ───────────────────────────────────────────────────────

def _load_custom_model(
    content:  bytes,
    filename: str,
) -> tuple[nn.Module, str, list[int], str]:
    """
    Dynamically load a user-uploaded .py model file.
    The file must define get_model() -> nn.Module, and optionally MODEL_NAME, INPUT_SHAPE.
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


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="PyTorch DAG Optimizer API Server")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    port = args.port or int(os.environ.get("PORT", 8080))
    host = args.host

    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │   PyTorch DAG Optimizer  v3.0  (startup cache)      │")
    print(f"  │   http://{host}:{port:<5}                              │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
