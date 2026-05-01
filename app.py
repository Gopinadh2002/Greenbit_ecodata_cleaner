import os
import threading
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from run_pipeline import stream_pipeline
from datetime import datetime
import tkinter as tk
from tkinter import filedialog


# Custom JSON provider for numpy types
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

app = Flask(
    __name__,
    static_folder='FrontEnd',
    static_url_path='/static',
    template_folder='FrontEnd'
)
app.json = NumpyJSONProvider(app)
CORS(app)

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
pipeline_state = {
    "running": False,
    "progress": 0,
    "status": "idle",
    "error": None,
    "results": None,
    "stats": None
}

# --- PAGE ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/regulations')
def regulations():
    return render_template('regulations.html')

# --- API ENDPOINTS ---
@app.route('/api/status')
def get_status():
    # Clean numpy types from pipeline_state
    clean_state = convert_numpy_types(pipeline_state)
    return jsonify(clean_state)

@app.route('/api/pick-folder', methods=['GET'])
def pick_folder():
    """Open native folder picker dialog and return selected folder path"""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the window
        root.attributes('-topmost', True)  # Bring to front

        folder_path = filedialog.askdirectory(
            title="Select a folder to scan",
            initialdir=os.path.expanduser("~")
        )

        root.destroy()

        if folder_path:
            return jsonify({"path": folder_path})
        else:
            return jsonify({"path": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline_api():
    global pipeline_state
    print(f"[API] /run-pipeline called, running={pipeline_state['running']}")

    if pipeline_state["running"]:
        print("[API] Pipeline already running")
        return jsonify({"error": "Pipeline already running"}), 409

    try:
        folder = request.json.get('folder', '.')
        print(f"[API] Received folder: {folder}")
        folder = os.path.normpath(folder)
        print(f"[API] Normalized folder: {folder}")

        if not os.path.isdir(folder):
            print(f"[API] Directory not found: {folder}")
            return jsonify({"error": f"Directory not found: {folder}"}), 404

        # Update existing dict instead of replacing it
        pipeline_state["running"] = True
        pipeline_state["progress"] = 0
        pipeline_state["status"] = "starting"
        pipeline_state["error"] = None
        pipeline_state["results"] = None
        pipeline_state["stats"] = None
        pipeline_state["folder"] = folder
        print(f"[API] Pipeline state updated to running")

        thread = threading.Thread(target=_execute_pipeline, args=(folder,))
        thread.daemon = True
        thread.start()
        print(f"[API] Background thread started")

        return jsonify({"status": "running", "folder": folder})
    except Exception as e:
        print(f"[API] ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

def _execute_pipeline(folder):
    global pipeline_state
    print(f"[PIPELINE] Starting pipeline for folder: {folder}")
    try:
        pipeline_state["status"] = "running"
        print(f"[PIPELINE] Calling stream_pipeline...")
        result_df, stats = stream_pipeline(folder)

        # Convert numpy types to Python native types for JSON serialization
        clean_stats = convert_numpy_types(stats)
        pipeline_state["stats"] = clean_stats
        pipeline_state["status"] = "complete"
        pipeline_state["progress"] = 100
        print(f"[PIPELINE] Pipeline completed successfully")
    except Exception as e:
        print(f"[PIPELINE] ERROR: {str(e)}")
        pipeline_state["error"] = str(e)
        pipeline_state["status"] = "error"
    finally:
        pipeline_state["running"] = False
        print(f"[PIPELINE] Pipeline finished, running=False")

@app.route('/api/results')
def get_results():
    if not pipeline_state["stats"]:
        return jsonify({"error": "No results"}), 404
    # Return clean stats
    clean_stats = convert_numpy_types(pipeline_state["stats"])
    return jsonify({"stats": clean_stats})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)