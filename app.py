#!/usr/bin/sage -python

# app.py - Web interface for gnn.py.
# Copyright (C) 2025 Soumendra Ganguly

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from flask import Flask, render_template_string, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import threading, time, os, json, re
import gnn  # Import the main module

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Geometry-Aware GNN Dashboard</title>
  <style>
    body { 
      font-family: Arial, sans-serif; 
      margin: 0; 
      padding: 2em; 
      background-color: #fafafa; 
      color: #333; 
    }
    h1 { color: #444; margin-bottom: 0.5em; }
    h2 { color: #555; margin-top: 2em; border-bottom: 2px solid #007bff; padding-bottom: 0.3em; }
    
    .controls {
      background: white;
      padding: 1.5em;
      border-radius: 10px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 2em;
    }
    
    button { 
      padding: 10px 20px; 
      border: none; 
      border-radius: 8px; 
      background-color: #007bff; 
      color: white; 
      cursor: pointer;
      font-size: 14px;
    }
    button:hover { background-color: #0056b3; }
    button:disabled { background-color: #ccc; cursor: not-allowed; }
    
    select, input { 
      padding: 8px; 
      border-radius: 5px; 
      border: 1px solid #ccc; 
      margin-right: 10px;
      font-size: 14px;
    }
    
    #log { 
      white-space: pre-wrap; 
      background-color: #2b2b2b; 
      color: #f0f0f0;
      padding: 15px; 
      border-radius: 10px; 
      height: 300px; 
      overflow-y: scroll; 
      font-family: 'Courier New', monospace;
      font-size: 12px;
    }
    
    .results-section {
      background: white;
      padding: 1.5em;
      border-radius: 10px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 2em;
    }
    
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1em;
      margin-top: 1em;
    }
    
    .summary-card {
      background: #f8f9fa;
      padding: 1em;
      border-radius: 8px;
      border-left: 4px solid #007bff;
    }
    
    .summary-card h3 {
      margin: 0 0 0.5em 0;
      color: #007bff;
      font-size: 1.1em;
    }
    
    .summary-card p {
      margin: 0.3em 0;
      font-size: 0.9em;
    }
    
    .images-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1.5em;
      margin-top: 1em;
    }
    
    .image-container {
      text-align: center;
    }
    
    .image-container img {
      width: 100%;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .image-container p {
      margin-top: 0.5em;
      font-size: 0.9em;
      color: #666;
    }
    
    .summary-plot {
      text-align: center;
      margin-top: 1em;
    }
    
    .summary-plot img {
      max-width: 100%;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .hidden { display: none; }
  </style>
</head>
<body>
  <h1>🧠 Geometry-Aware GNN Experiment Dashboard</h1>
  
  <div class="controls">
    <form id="runForm">
      <label><strong>Test Geometry:</strong></label>
      <select id="geom">
        <option value="euclidean">Euclidean</option>
        <option value="spherical">Spherical</option>
        <option value="hyperbolic">Hyperbolic</option>
      </select>
      
      <label><strong>Runs:</strong></label>
      <input type="number" id="runs" value="3" min="1" max="10">
      
      <button type="submit" id="runBtn">Run Experiment</button>
    </form>
  </div>

  <div class="results-section">
    <h2>📊 Experiment Log</h2>
    <div id="log"></div>
  </div>

  <div id="summarySection" class="results-section hidden">
    <h2>📈 Summary Statistics</h2>
    <div id="summaryStats" class="summary-grid"></div>
    <div id="summaryPlot" class="summary-plot"></div>
  </div>

  <div id="vizSection" class="results-section hidden">
    <h2>🎨 Graph Visualizations</h2>
    <div id="vizContainer"></div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
  <script>
    const socket = io();
    const form = document.getElementById('runForm');
    const runBtn = document.getElementById('runBtn');
    const log = document.getElementById('log');
    const summarySection = document.getElementById('summarySection');
    const summaryStats = document.getElementById('summaryStats');
    const summaryPlot = document.getElementById('summaryPlot');
    const vizSection = document.getElementById('vizSection');
    const vizContainer = document.getElementById('vizContainer');

    form.onsubmit = (e) => {
      e.preventDefault();
      
      // Reset UI
      log.textContent = '';
      summarySection.classList.add('hidden');
      vizSection.classList.add('hidden');
      summaryStats.innerHTML = '';
      summaryPlot.innerHTML = '';
      vizContainer.innerHTML = '';
      
      // Disable button
      runBtn.disabled = true;
      runBtn.textContent = 'Running...';
      
      const geom = document.getElementById('geom').value;
      const runs = document.getElementById('runs').value;
      socket.emit('run_experiment', { geom, runs });
    };

    socket.on('log', (msg) => {
      log.textContent = log.textContent + msg + '\\n';
      log.scrollTop = log.scrollHeight;
    });

    socket.on('result', (data) => {
      runBtn.disabled = false;
      runBtn.textContent = 'Run Experiment';
      
      // Display summary statistics
      summarySection.classList.remove('hidden');
      const summary = data.summary;
      
      for (const [geom, stats] of Object.entries(summary)) {
        const card = document.createElement('div');
        card.className = 'summary-card';
        card.innerHTML = `
          <h3>${geom.charAt(0).toUpperCase() + geom.slice(1)}</h3>
          <p><strong>Accuracy:</strong> ${stats.acc_mean.toFixed(4)} ± ${stats.acc_std.toFixed(4)}</p>
          <p><strong>Time:</strong> ${stats.time_mean.toFixed(1)} ± ${stats.time_std.toFixed(1)}s</p>
        `;
        summaryStats.appendChild(card);
      }
      
      // Display summary plot
      if (data.summary_plot) {
        summaryPlot.innerHTML = `<img src="/${data.summary_plot}" alt="Summary Plot">`;
      }
      
      // Display visualizations
      if (data.visualizations && data.visualizations.length > 0) {
        vizSection.classList.remove('hidden');
        
        // Group by run number
        const runs = parseInt(document.getElementById('runs').value);
        for (let i = 1; i <= runs; i++) {
          const runDiv = document.createElement('div');
          runDiv.innerHTML = `<h3>Run ${i}</h3>`;
          const grid = document.createElement('div');
          grid.className = 'images-grid';
          
          const runImages = data.visualizations.filter(v => v.includes('-' + i + '.png'));
          runImages.forEach(imgPath => {
            const filename = imgPath.split('/').pop();
            const geomMatch = filename.match(/^(\\w+)_GNN/);
            const geomName = geomMatch ? geomMatch[1].charAt(0).toUpperCase() + geomMatch[1].slice(1) : '';
            
            const container = document.createElement('div');
            container.className = 'image-container';
            container.innerHTML = `
              <img src="/${imgPath}" alt="${filename}">
              <p>${geomName} GNN</p>
            `;
            grid.appendChild(container);
          });
          
          runDiv.appendChild(grid);
          vizContainer.appendChild(runDiv);
        }
      }
    });
  </script>
</body>
</html>"""

# --- Background Experiment Thread ---
def background_experiment(geom, runs):
    import builtins
    logs = []
    original_print = builtins.print
    save_dir = None
    
    def log_capture(*args, **kwargs):
        message = ' '.join(str(a) for a in args)
        logs.append(message)
        socketio.emit('log', message)
        original_print(*args, **kwargs)
    
    builtins.print = log_capture

    try:
        socketio.emit('log', f"Starting {runs} runs on {geom} geometry...")
        start = time.time()
        
        # Run with save_outputs=True to generate visualizations
        results = gnn.run_multiple(test_geom=geom, runs=int(runs), verbose=True, save_outputs=True)
        
        elapsed = time.time() - start
        socketio.emit('log', f"All runs completed in {elapsed:.1f}s.")
        
        # Extract save_dir from logs
        for log_msg in logs:
            if 'Created output directory:' in log_msg:
                save_dir = log_msg.split('Created output directory:')[1].strip()
                break
        
        # Collect visualization paths
        visualizations = []
        summary_plot = None
        
        if save_dir and os.path.exists(save_dir):
            for filename in sorted(os.listdir(save_dir)):
                file_path = os.path.join(save_dir, filename)
                if filename.endswith('.png'):
                    if filename == 'multi_run_results.png':
                        summary_plot = file_path
                    else:
                        visualizations.append(file_path)
        
        # Send results with paths
        socketio.emit('result', {
            'summary': results,
            'visualizations': visualizations,
            'summary_plot': summary_plot,
            'save_dir': save_dir
        })
        
    finally:
        builtins.print = original_print

# --- Socket Events ---
@socketio.on('run_experiment')
def handle_run_experiment(data):
    geom = data.get('geom', 'euclidean')
    runs = data.get('runs', 3)
    thread = threading.Thread(target=background_experiment, args=(geom, runs))
    thread.start()

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/results_<path:filename>')
def serve_results(filename):
    """Serve files from results directories"""
    directory = f'results_{filename.split("/")[0].split("_")[0]}_{filename.split("/")[0].split("_", 1)[1]}'
    file = filename.split('/', 1)[1] if '/' in filename else filename
    return send_from_directory(directory, file)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
