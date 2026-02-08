#!/usr/bin/env python3
"""
ANANKE Web UI - Flask front-end for SAM/ANANKE dual system.
"""

import time
from flask import Flask, jsonify, request, render_template_string

try:
    import sam_ananke_dual_system as ananke_module
except ImportError:
    ananke_module = None

APP = Flask(__name__)
ARENA = ananke_module.create(16, 4, 42) if ananke_module else None

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ANANKE Control</title>
  <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #0b0e14; color: #e6edf3; margin: 0; }
    header { padding: 16px 24px; background: #111827; border-bottom: 1px solid #1f2937; }
    .grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); padding: 24px; }
    .card { background: #111827; padding: 16px; border: 1px solid #1f2937; border-radius: 10px; }
    button { background: #2563eb; color: #fff; border: 0; padding: 10px 14px; border-radius: 8px; cursor: pointer; }
    button.secondary { background: #374151; }
  </style>
</head>
<body>
  <header>
    <h2>ANANKE Dual System</h2>
    <div>Live SAM/ANANKE arena metrics</div>
  </header>
  <section class="grid">
    <div class="card">
      <h3>State</h3>
      <pre id="state">loading...</pre>
    </div>
    <div class="card">
      <h3>Control</h3>
      <button onclick="step(1)">Step</button>
      <button class="secondary" onclick="step(10)">Step x10</button>
      <button class="secondary" onclick="step(100)">Step x100</button>
    </div>
  </section>
  <script>
    async function fetchState() {
      const res = await fetch('/api/state');
      const data = await res.json();
      document.getElementById('state').textContent = JSON.stringify(data, null, 2);
    }
    async function step(n) {
      await fetch('/api/step', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({steps:n})});
      await fetchState();
    }
    setInterval(fetchState, 2000);
    fetchState();
  </script>
</body>
</html>
"""

@APP.route("/")
def index():
    return render_template_string(INDEX_HTML)

@APP.route("/api/state")
def api_state():
    if not ananke_module or not ARENA:
        return jsonify({"error": "ananke unavailable"}), 503
    return jsonify(ananke_module.get_state(ARENA))

@APP.route("/api/step", methods=["POST"])
def api_step():
    if not ananke_module or not ARENA:
        return jsonify({"error": "ananke unavailable"}), 503
    payload = request.get_json(silent=True) or {}
    steps = int(payload.get("steps", 1))
    if steps < 1:
        steps = 1
    ananke_module.run(ARENA, steps)
    return jsonify(ananke_module.get_state(ARENA))

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5006, debug=False)
