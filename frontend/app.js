/**
 * app.js — PyTorch DAG Optimizer Frontend
 *
 * Responsibilities:
 *  - Fetch built-in model list from /api/models
 *  - Upload / select a model and POST to /api/analyze
 *  - Render vis-network graph from the API response
 *  - Update metrics sidebar dynamically
 *  - Manage loading states with multi-step status indicators
 *  - Show node metadata on click
 */

// ── Constants ──────────────────────────────────────────────────────────────

// Node colour palette (matches style.css)
const COLORS = {
  placeholder:   { bg: '#a78bfa', border: '#7c3aed', font: '#0b0e17' },
  call_module:   { bg: '#1e3a5f', border: '#4f8ef7', font: '#c7dcff' },
  call_function: { bg: '#3d2800', border: '#f59e0b', font: '#fef3c7' },
  fused_op:      { bg: '#003d30', border: '#00ffc8', font: '#00ffc8' },
  folded_op:     { bg: '#1a3320', border: '#4ade80', font: '#bbf7d0' },
  output:        { bg: '#4a0030', border: '#f472b6', font: '#fce7f3' },
  default:       { bg: '#1a2235', border: '#334155', font: '#e2e8f0' },
};

// ── State ──────────────────────────────────────────────────────────────────

let network      = null;
let nodesDS      = null;
let edgesDS      = null;
let activeKey    = null;     // 'original' | 'optimized'
let selectedFile = null;     // File object from drag-and-drop or browse
let analysisData = null;     // Full API response (both graphs + metrics)
let downloadUrls = null;     // { pt_url, json_url } from last analysis

// ── Utilities ──────────────────────────────────────────────────────────────

function el(id) { return document.getElementById(id); }

function humanBytes(b) {
  if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(1) + ' KB';
  return b + ' B';
}

function nodeColour(node) {
  if (node.op_type === 'fused_op')  return COLORS.fused_op;
  if (node.op_type === 'folded_op') return COLORS.folded_op;
  return COLORS[node.op_type] ?? COLORS.default;
}

function fusedGlow(node) {
  if (node.op_type !== 'fused_op') return {};
  return {
    shadow: {
      enabled: true,
      color:   'rgba(0,255,200,0.6)',
      size:    20,
      x: 0, y: 0,
    },
  };
}

// ── Graph builder ──────────────────────────────────────────────────────────

function buildDatasets(graphData) {
  const visNodes = graphData.nodes.map(n => {
    const clr  = nodeColour(n);
    const isFused  = n.op_type === 'fused_op';
    const isFolded = n.op_type === 'folded_op';

    let label;
    if (isFused) {
      const key = n.node_id
        .replace(/^fused_cbr_/, '')
        .replace(/_conv\d+$/, '')
        .replace(/_/g, '.');
      label = `⚡ Conv·BN·ReLU\n${key}`;
    } else if (isFolded) {
      const key = n.node_id
        .replace(/^folded_cb_/, '')
        .replace(/_/g, '.');
      label = `🔗 Conv·BN\n${key}`;
    } else {
      label = `${n.name}\n${n.op_type}`;
    }

    return {
      id:    n.node_id,
      label,
      title: buildTooltip(n),
      shape: isFused ? 'hexagon' : isFolded ? 'hexagon' : (n.op_type === 'placeholder' ? 'diamond' : 'box'),
      color: {
        background:  clr.bg,
        border:      clr.border,
        highlight:   { background: clr.border, border: '#ffffff' },
        hover:       { background: clr.border, border: '#ffffff' },
      },
      font: {
        color:    clr.font,
        size:     (isFused || isFolded) ? 12 : 11,
        face:     'JetBrains Mono, monospace',
        multi:    'html',
      },
      borderWidth:      (isFused || isFolded) ? 2.5 : 1.5,
      borderWidthSelected: 3,
      size:             (isFused || isFolded) ? 32 : 18,
      widthConstraint:  (isFused || isFolded) ? { minimum: 100, maximum: 115 } : undefined,
      ...fusedGlow(n),
      _raw: n,
    };
  });

  const visEdges = graphData.edges.map((e, i) => ({
    id:     `e${i}`,
    from:   e.src,
    to:     e.dst,
    arrows: { to: { enabled: true, scaleFactor: 0.6 } },
    color:  { color: 'rgba(100,116,139,0.5)', highlight: '#6366f1', hover: '#818cf8' },
    width:  1.2,
    smooth: { type: 'curvedCW', roundness: 0.08 },
  }));

  return { visNodes, visEdges };
}

function buildTooltip(n) {
  const shape = n.output_shape ? n.output_shape.join(' × ') : 'N/A';
  const klass = n.kwargs?.module_class ?? '';
  const fused = n.kwargs?.fused_from;

  let html = `<div style="
      background:#0f172a; border:1px solid #334155; border-radius:8px;
      padding:10px 14px; font-family:'JetBrains Mono',monospace;
      font-size:11px; color:#cbd5e1; max-width:260px; line-height:1.7;">
    <b style="color:#e2e8f0;font-size:12px">${n.name}</b><br/>
    <span style="color:#64748b">op_type: </span><span style="color:#818cf8">${n.op_type}</span><br/>
    <span style="color:#64748b">shape:   </span><span style="color:#a5f3fc">${shape}</span>`;
  if (klass) html += `<br/><span style="color:#64748b">class:   </span><span>${klass}</span>`;
  if (fused) html += `<br/><span style="color:#64748b">fused:   </span><span style="color:#00ffc8">${fused.join(' + ')}</span>`;
  html += '</div>';
  return html;
}

// ── Metrics sidebar ────────────────────────────────────────────────────────

function updateMetricsFromAPI(metrics) {
  const nodes  = metrics.optimized_nodes;
  const edges  = metrics.optimized_edges;
  const fusedN = metrics.fused_ops;

  // ── Core counts ──────────────────────────────────────────────────────────
  el('m-nodes').textContent = nodes;
  el('m-edges').textContent = edges;
  el('m-fused').textContent = fusedN > 0 ? `${fusedN} chains` : '0';

  // ── R/W elimination ───────────────────────────────────────────────────────
  el('m-rw').textContent = metrics.hbm_rw_eliminated > 0
    ? `${metrics.hbm_rw_eliminated} ops eliminated` : 'N/A';
  el('m-bw').textContent = metrics.hbm_bandwidth_freed_bytes > 0
    ? `${metrics.hbm_bandwidth_freed_human} freed` : 'N/A';

  // ── Significance measurements ─────────────────────────────────────────────
  setSignificance('sig-nodes', `${metrics.node_reduction_pct}%`,
    metrics.node_reduction_pct, 60,
    'Node count reduction vs original graph');
  setSignificance('sig-edges', `${metrics.edge_reduction_pct}%`,
    metrics.edge_reduction_pct, 60,
    'Edge count reduction vs original graph');
  setSignificance('sig-bw', `${metrics.hbm_traffic_cut_pct}%`,
    metrics.hbm_traffic_cut_pct, 40,
    'HBM read/write traffic eliminated vs total original edge traversals');
  setSignificance('sig-tput', `~${metrics.est_throughput_gain_pct}%`,
    metrics.est_throughput_gain_pct, 30,
    'Est. throughput gain (empirical: ~1.8% per Conv-BN-ReLU fusion, conservative)');
}

function updateMetricsForGraph(graphData, key) {
  // When toggling between original/optimized, update the core counts
  const nodes  = graphData.metadata.num_nodes;
  const edges  = graphData.metadata.num_edges;
  const fusedN = graphData.nodes.filter(n => n.op_type === 'fused_op').length;

  el('m-nodes').textContent = nodes;
  el('m-edges').textContent = edges;
  el('m-fused').textContent = fusedN > 0 ? `${fusedN} chains` : '0';

  // Footer
  el('footer-graph-name').textContent = graphData.metadata.graph_name;
  const ts = graphData.metadata.optimized_at ?? graphData.metadata.created_at ?? '';
  el('footer-created').textContent = ts ? `Generated: ${ts}` : '';
}

/**
 * Render a significance bar into element #id.
 */
function setSignificance(id, label, value, maxVal, tooltip) {
  const el_ = el(id);
  if (!el_) return;
  const pct   = Math.min(100, (value / maxVal) * 100);
  const color = value >= 15 ? '#4ade80' : value >= 8 ? '#f59e0b' : '#64748b';
  el_.title = tooltip;
  el_.innerHTML = `
    <div class="sig-header">
      <span class="sig-label">${el_.dataset.name ?? ''}</span>
      <span class="sig-val" style="color:${color}">${label}</span>
    </div>
    <div class="sig-track">
      <div class="sig-fill" style="width:${pct}%;background:${color}"></div>
    </div>`;
}

// ── Pipeline panel ─────────────────────────────────────────────────────────

function renderPipeline(passResults) {
  const body = el('pipeline-body');
  if (!body || !passResults) return;

  body.innerHTML = passResults.map(r => {
    const deltaN = r.nodes_before - r.nodes_after;
    const deltaE = r.edges_before - r.edges_after;
    const deltaStr = deltaN > 0
      ? `<span class="pipeline-item-delta">−${deltaN} nodes, −${deltaE} edges</span>`
      : `<span class="pipeline-item-delta" style="color:var(--text-dim)">no change</span>`;

    return `<div class="pipeline-item">
      <div class="pipeline-item-header">
        <span class="pipeline-item-name">${r.pass_name}</span>
        ${deltaStr}
      </div>
      <div class="pipeline-item-stats">${r.nodes_before} → ${r.nodes_after} nodes | ${r.edges_before} → ${r.edges_after} edges</div>
    </div>`;
  }).join('');

  el('pipeline-panel').style.display = 'block';
}

// ── Node click handler ─────────────────────────────────────────────────────

function showNodeInfo(nodeId) {
  const nodeData = nodesDS.get(nodeId);
  if (!nodeData) return;
  const n = nodeData._raw;
  if (!n) return;

  const isFused = n.op_type === 'fused_op' || n.op_type === 'folded_op';
  const shape   = n.output_shape ? `[${n.output_shape.join(', ')}]` : 'N/A';
  const klass   = n.kwargs?.module_class ?? '—';
  const fused   = n.kwargs?.fused_from;

  let html = `
    <div class="ni-row"><span class="ni-key">id:</span><span class="ni-val ${isFused ? 'ni-fused' : ''}">${n.node_id}</span></div>
    <div class="ni-row"><span class="ni-key">op:</span><span class="ni-val">${n.op_type}</span></div>
    <div class="ni-row"><span class="ni-key">name:</span><span class="ni-val">${n.name}</span></div>
    <div class="ni-row"><span class="ni-key">shape:</span><span class="ni-val">${shape}</span></div>
    <div class="ni-row"><span class="ni-key">class:</span><span class="ni-val">${klass}</span></div>
  `;
  if (fused) {
    html += `<div class="ni-row" style="flex-direction:column;gap:2px">
      <span class="ni-key">fused from:</span>
      ${fused.map(f => `<span class="ni-val ni-fused">• ${f}</span>`).join('')}
    </div>`;
  }
  if (n.kwargs?.fusion_notes) {
    html += `<div class="ni-row" style="flex-direction:column;gap:4px;margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,.06)">
      <span class="ni-key">notes:</span>
      <span style="color:#94a3b8;font-size:10px;line-height:1.5">${n.kwargs.fusion_notes}</span>
    </div>`;
  }
  if (n.kwargs?.fold_notes) {
    html += `<div class="ni-row" style="flex-direction:column;gap:4px;margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,.06)">
      <span class="ni-key">notes:</span>
      <span style="color:#94a3b8;font-size:10px;line-height:1.5">${n.kwargs.fold_notes}</span>
    </div>`;
  }

  el('node-info-body').innerHTML = html;
  el('node-info-panel').style.display = 'block';
}

// ── Graph renderer ─────────────────────────────────────────────────────────

function renderGraph(graphData) {
  const { visNodes, visEdges } = buildDatasets(graphData);

  nodesDS = new vis.DataSet(visNodes);
  edgesDS = new vis.DataSet(visEdges);

  const container = el('graph-canvas');

  const options = {
    layout: {
      hierarchical: {
        enabled:          true,
        direction:        'UD',
        sortMethod:       'directed',
        levelSeparation:  90,
        nodeSpacing:      120,
        treeSpacing:      140,
        blockShifting:    true,
        edgeMinimization: true,
        parentCentralization: true,
      },
    },
    interaction: {
      hover:          true,
      tooltipDelay:   120,
      navigationButtons: false,
      keyboard:       false,
      zoomView:       true,
    },
    physics: { enabled: false },
    nodes: {
      margin: 8,
      widthConstraint: { minimum: 90, maximum: 170 },
    },
    edges: {
      hoverWidth: 2.5,
      selectionWidth: 2.5,
    },
  };

  if (network) {
    network.destroy();
  }
  network = new vis.Network(container, { nodes: nodesDS, edges: edgesDS }, options);

  network.on('click', params => {
    if (params.nodes.length > 0) {
      showNodeInfo(params.nodes[0]);
      el('canvas-hint').style.display = 'none';
    } else {
      el('node-info-panel').style.display = 'none';
      el('canvas-hint').style.display = 'block';
    }
  });

  // Fit after stabilization
  network.once('afterDrawing', () => {
    network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
  });
}

// ── Loading overlay management ─────────────────────────────────────────────

function showLoading(message) {
  el('loading-overlay').style.display = 'flex';
  el('loading-message').textContent = message || 'Processing…';
  el('canvas-welcome').style.display = 'none';
  // Reset all steps
  ['step-upload', 'step-trace', 'step-optimize', 'step-metrics'].forEach(id => {
    el(id).className = 'status-step';
  });
}

function setStep(stepId) {
  const steps = ['step-upload', 'step-trace', 'step-optimize', 'step-metrics'];
  const idx = steps.indexOf(stepId);
  steps.forEach((id, i) => {
    if (i < idx)       el(id).className = 'status-step done';
    else if (i === idx) el(id).className = 'status-step active';
    else                el(id).className = 'status-step';
  });
}

function hideLoading() {
  el('loading-overlay').style.display = 'none';
}

// ── Error toast ────────────────────────────────────────────────────────────

let errorTimeout = null;

function showError(message) {
  // Create or reuse toast
  let toast = document.querySelector('.error-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.className = 'error-toast';
    document.body.appendChild(toast);
  }
  toast.textContent = `⚠ ${message}`;
  // Force reflow then show
  toast.classList.remove('show');
  void toast.offsetWidth;
  toast.classList.add('show');

  if (errorTimeout) clearTimeout(errorTimeout);
  errorTimeout = setTimeout(() => toast.classList.remove('show'), 5000);
}

// ── Upload & Analysis ──────────────────────────────────────────────────────

async function fetchModels() {
  try {
    const res = await fetch('/api/models');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const select = el('model-select');
    select.innerHTML = '<option value="" disabled selected>Choose a model…</option>';
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = `${m.display_name} — ${m.description}`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error('Failed to fetch models:', err);
    el('model-select').innerHTML = '<option value="" disabled selected>Could not load models</option>';
  }
}

function updateAnalyzeButton() {
  const hasModel = el('model-select').value !== '';
  const hasFile  = selectedFile !== null;
  el('analyze-btn').disabled = !(hasModel || hasFile);
}

async function analyzeModel() {
  const btn = el('analyze-btn');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.innerHTML = '<span class="analyze-btn-icon">⏳</span> Analyzing…';

  showLoading('Analyzing model architecture…');
  setStep('step-upload');

  el('status-badge').textContent = '● ANALYZING';
  el('status-badge').style.color = '#f59e0b';

  try {
    const formData = new FormData();

    if (selectedFile) {
      formData.append('file', selectedFile);
    } else {
      formData.append('model_name', el('model-select').value);
    }

    // Simulate step progression (the backend does it all at once,
    // but we animate the steps for UX)
    setStep('step-upload');
    await sleep(400);
    setStep('step-trace');

    const res = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });

    setStep('step-optimize');
    await sleep(300);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.detail || 'Unknown error');

    setStep('step-metrics');
    await sleep(300);

    // ── Store analysis data ─────────────────────────────────────────────
    analysisData = data;
    downloadUrls = data.downloads || {};

    // ── Wire up download buttons ─────────────────────────────────────────
    const ptBtn = el('download-pt-btn');
    if (ptBtn) {
      ptBtn.disabled = !downloadUrls.pt_url;
    }

    // ── Show results panels ─────────────────────────────────────────────
    el('graph-source-panel').style.display = 'block';
    el('metrics-panel').style.display      = 'block';
    el('impact-panel').style.display       = 'block';

    // Update button tags with actual node counts
    el('tag-original').textContent  = `${data.metrics.original_nodes} nodes`;
    el('tag-optimized').textContent = `${data.metrics.optimized_nodes} nodes`;

    // ── Render the optimized graph by default ────────────────────────────
    activeKey = null;
    loadGraph('optimized');

    // ── Update all metrics from API response ────────────────────────────
    updateMetricsFromAPI(data.metrics);
    renderPipeline(data.metrics.pass_results);

    // ── Footer ──────────────────────────────────────────────────────────
    el('footer-model-name').textContent = data.model_name;
    el('footer-graph-name').textContent = data.optimized_graph.metadata.graph_name;
    const ts = data.optimized_graph.metadata.optimized_at ?? '';
    el('footer-created').textContent = ts ? `Generated: ${ts}` : '';

    el('status-badge').textContent = '● READY';
    el('status-badge').style.color = '#4ade80';

  } catch (err) {
    console.error('Analysis failed:', err);
    showError(err.message);
    el('status-badge').textContent = '● ERROR';
    el('status-badge').style.color = '#ef4444';
  } finally {
    hideLoading();
    btn.classList.remove('loading');
    btn.innerHTML = '<span class="analyze-btn-icon">⚡</span> Analyze & Optimize';
    updateAnalyzeButton();
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Graph loader (now reads from analysisData in memory) ───────────────────

async function loadGraph(key) {
  if (key === activeKey && network) return;
  activeKey = key;

  // Button state
  el('btn-original').classList.toggle('btn-active',  key === 'original');
  el('btn-optimized').classList.toggle('btn-active', key === 'optimized');

  if (!analysisData) return;

  const graphData = key === 'original'
    ? analysisData.original_graph
    : analysisData.optimized_graph;

  const overlay = el('loading-overlay');
  overlay.style.display = 'flex';
  el('loading-message').textContent = 'Rendering graph…';
  // Hide status steps for simple graph switch
  el('status-steps').style.display = 'none';

  el('status-badge').textContent = '● LOADING';
  el('status-badge').style.color = '#f59e0b';

  try {
    renderGraph(graphData);
    updateMetricsForGraph(graphData, key);

    el('status-badge').textContent = '● READY';
    el('status-badge').style.color = '#4ade80';
    el('canvas-hint').style.display = 'block';
  } catch (err) {
    console.error('Failed to render graph:', err);
    showError('Failed to render graph');
    el('status-badge').textContent = '● ERROR';
    el('status-badge').style.color = '#ef4444';
  } finally {
    overlay.style.display = 'none';
    el('status-steps').style.display = 'flex';
  }
}

// ── Drag-and-drop handlers ─────────────────────────────────────────────────

function initUploadZone() {
  const zone  = el('upload-zone');
  const input = el('file-input');

  // Click to browse
  zone.addEventListener('click', () => input.click());

  // File selected via browse
  input.addEventListener('change', () => {
    if (input.files.length > 0) {
      selectFile(input.files[0]);
    }
  });

  // Drag events
  zone.addEventListener('dragenter', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
  });
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      selectFile(e.dataTransfer.files[0]);
    }
  });

  // Clear file button
  el('upload-clear-btn').addEventListener('click', e => {
    e.stopPropagation();
    clearFile();
  });

  // Model select change
  el('model-select').addEventListener('change', () => {
    // If user picks a built-in model, clear any uploaded file
    clearFile();
    updateAnalyzeButton();
  });

  // Analyze button
  el('analyze-btn').addEventListener('click', analyzeModel);
}

function selectFile(file) {
  if (!file.name.endsWith('.py')) {
    showError('Only .py files are accepted');
    return;
  }
  selectedFile = file;
  el('upload-zone').style.display = 'none';
  el('upload-filename').style.display = 'flex';
  el('upload-filename-text').textContent = file.name;
  // Clear model select since file takes precedence
  el('model-select').value = '';
  updateAnalyzeButton();
}

function clearFile() {
  selectedFile = null;
  el('file-input').value = '';
  el('upload-zone').style.display = 'block';
  el('upload-filename').style.display = 'none';
  updateAnalyzeButton();
}
// ── Download optimized model (server JSON or in-memory fallback) ──────────

function downloadOptimizedGraph() {
  if (!analysisData) {
    showError('No analysis available — run an analysis first.');
    return;
  }

  // Prefer the server-side JSON (richer, includes memory analysis)
  if (downloadUrls && downloadUrls.json_url) {
    const a = document.createElement('a');
    a.href     = downloadUrls.json_url;
    a.download = downloadUrls.json_filename || 'optimized_dag.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    return;
  }

  // Fallback: build from in-memory data
  if (!analysisData.optimized_graph) {
    showError('No optimized graph available.');
    return;
  }
  const exportData = {
    _export_info: {
      tool:        'PyTorch DAG Optimizer',
      exported_at: new Date().toISOString(),
      description: 'Optimized computation graph — Kahn BFS topo-sort, fused/folded operators.',
    },
    model_name:       analysisData.model_name,
    optimized_graph:  analysisData.optimized_graph,
    metrics:          analysisData.metrics,
    memory_analysis:  analysisData.memory_analysis,
  };
  const json = JSON.stringify(exportData, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const safeName = (analysisData.model_name || 'model')
    .toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
  const a = document.createElement('a');
  a.href     = url;
  a.download = `${safeName}_optimized_dag.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Download TorchScript .pt model ─────────────────────────────────────────

function downloadPtModel() {
  if (!downloadUrls || !downloadUrls.pt_url) {
    showError('TorchScript model not available — run an analysis first.');
    return;
  }
  const a = document.createElement('a');
  a.href     = downloadUrls.pt_url;
  a.download = downloadUrls.pt_filename || 'optimized_model.pt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ── Boot ───────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  initUploadZone();
  fetchModels();
});
