# ⬡ PyTorch DAG Optimizer

> **Data Structures Bonus Project** — ECE Course, 2026

An interactive web application that traces, visualizes, and **automatically optimizes** PyTorch neural network computation graphs using graph algorithms and operator fusion techniques.

🔗 **Live Demo**: [https://pytorch-dag-optimizer.onrender.com](https://pytorch-dag-optimizer.onrender.com) *(free tier — wakes in ~20 seconds)*

---

## What It Does

Upload any PyTorch model (`.py` file) or select a built-in architecture. The app:

1. **Traces** the model using `torch.fx` to extract a computation DAG
2. **Topologically sorts** the graph using **Kahn's BFS algorithm**
3. **Runs a multi-pass optimization pipeline**:
   - Conv-BN-ReLU Fusion — merges 3-node chains into 1 fused operator
   - Dead Node Elimination — BFS reachability, removes unreachable nodes
   - Conv-BN Folding — mathematically merges Conv weights into BN
   - Memory Footprint Analysis — annotates peak activation memory
4. **Visualizes** both the original and optimized graphs interactively
5. **Exports** an **Optimized DAG JSON** — the graph structure for visualization, research, or C++ LibTorch integration

---

## Why It's Fast (Technical Explanation)

| Optimization | Effect |
|---|---|
| **Conv-BN Folding** | BN weights absorbed into Conv weights → one fewer layer at inference |
| **TorchScript Compilation** | Removes Python GIL overhead, enables kernel-level fusion |
| **HBM Read/Write Elimination** | Fused ops skip intermediate memory writes → lower memory bandwidth |
| **Kahn's BFS Topo-Sort** | Optimal node ordering for cache-efficient forward passes |

The optimization passes (Conv-BN folding, operator fusion) apply real mathematical transformations to the graph. When running locally, you can also export a TorchScript `.pt` model for production inference.

---

## Project Structure

```
PyTorch_DAG/
├── src/
│   ├── dag_core.py          # Custom DAG data structure (nodes, edges, adjacency)
│   ├── graph_extractor.py   # torch.fx tracer → DAG
│   ├── optimizer_passes.py  # 4 optimization passes (PassManager pattern)
│   └── optimizer_agent.py   # Pipeline orchestration + CLI entry point
├── frontend/
│   ├── index.html           # Single-page app
│   ├── app.js               # vis-network graph renderer + API client
│   └── style.css            # Dark theme design system
├── precomputed/             # Pre-generated analysis results (committed to repo)
├── downloads/               # Pre-generated JSON exports for download
├── api.py                   # FastAPI server (serves pre-computed results)
├── precompute.py            # Offline analysis generator (run locally)
├── sample_model.py          # Example custom model for upload
├── requirements.txt         # CPU-only PyTorch + FastAPI + gunicorn
└── Procfile                 # Render.com deployment config
```


---

## Data Structures Used

| Structure | Where | Purpose |
|---|---|---|
| **DAG (Directed Acyclic Graph)** | `dag_core.py` | Models computation flow |
| **Adjacency list** (dict of lists) | `optimizer_agent.py` | Forward/reverse edges for O(V+E) traversal |
| **BFS queue** | `dag_core.py` | Kahn's topological sort |
| **Hash map** (dict) | `graph_extractor.py` | Node ID → node data, O(1) lookup |
| **Pass pipeline** (list of objects) | `optimizer_passes.py` | Composable, ordered optimization passes |

---

## Running Locally

```bash
# 1. Create conda environment with PyTorch
conda create -n ai_env python=3.11
conda activate ai_env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python api.py

# 4. Open browser
open http://localhost:8080
```

---

## Uploading a Custom Model

Your `.py` file must define a `get_model()` function:

```python
import torch.nn as nn

MODEL_NAME  = "MyModel"        # optional
INPUT_SHAPE = [1, 3, 224, 224] # optional

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def get_model():
    return MyModel()
```

Download [`sample_model.py`](sample_model.py) as a working starting point.

---

## Using the Exported `.pt` Model

```python
import torch

# Load the TorchScript model (no original Python code needed)
model = torch.jit.load('optimized_resnet18.pt')
model.eval()

# Run inference
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(x)
print(output.shape)  # → torch.Size([1, 1000])
```

---

## Deployment (Render.com)

Deployed automatically from GitHub via [Render.com](https://render.com).

- **Build**: `pip install -r requirements.txt`
- **Start**: `gunicorn api:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`
- **Free tier** — no credit card required

---

## Screenshots

> Run the app and analyze ResNet-18 to see the optimizer in action:
> - Original graph: ~70+ nodes
> - Optimized graph: node count reduced by ~30-40%
> - Fused Conv-BN-ReLU chains glowing in teal ✦

---

*Built as a bonus project for the Data Structures course, demonstrating real-world applications of DAGs, BFS, and graph optimization algorithms.*
