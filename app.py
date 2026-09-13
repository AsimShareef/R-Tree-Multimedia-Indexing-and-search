import os
import numpy as np
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading # <-- NEW
import uuid      # <-- NEW

# Import your custom modules
from src.evaluation.visualizer import get_rtree_visualization
from src.evaluation.benchmark import get_benchmark_data
from src.search.execution import execute_user_query

app = Flask(__name__)
CORS(app) 

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
AUDIO_FOLDER = os.path.join(BASE_DIR, 'data', 'raw_audio')
IMAGE_FOLDER = os.path.join(BASE_DIR, 'data', 'raw_images')

# ==========================================
# PAGE ROUTES (Frontend Serving)
# ==========================================

BENCHMARK_JOBS = {}

def run_benchmark_background(job_id, n, k):
    """Background worker that runs the heavy computation."""
    try:
        # Run your heavy script
        benchmark_results = get_benchmark_data(N=n, k=k)
        # Store results when finished
        BENCHMARK_JOBS[job_id] = {"status": "Complete", "data": benchmark_results}
    except Exception as e:
        BENCHMARK_JOBS[job_id] = {"status": "Error", "error": str(e)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def render_search():
    return render_template('search.html')

@app.route('/visualizer_app')
def render_visualizer():
    return render_template('visualizer.html')

@app.route('/benchmark')
def render_benchmark():
    return render_template('benchmark.html')


# ==========================================
# API ROUTES (Backend Logic)
# ==========================================

# --- Module 1: KNN Search ---
@app.route('/api/search', methods=['POST'])
def search_api():
    try:
        media_type = request.form.get('mediaType')
        k_value = int(request.form.get('kValue'))
        uploaded_file = request.files['queryFile']

        if not uploaded_file:
            return jsonify({'error': 'No file uploaded'}), 400

        temp_path = f"temp_{secure_filename(uploaded_file.filename)}"
        uploaded_file.save(temp_path)

        results = execute_user_query(temp_path, media_type, k=k_value)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({'status': 'success', 'results': results})

    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

# --- Module 2: 2D Spatial Visualizer API ---
@app.route('/api/visualize_data', methods=['POST'])
def handle_visualize_data():
    """Generates random points for the browser's 2D Javascript visualizer."""
    data = request.json or {}
    num_points = data.get('num_points', 30)
    tree_type = data.get('tree_type', 'kdtree')
    
    np.random.seed(42) # Fixed seed for stable visual
    points = (np.random.rand(num_points, 2) * 100).tolist()
    
    return jsonify({"points": points, "tree_type": tree_type})

# --- Module 3: Benchmark API ---
@app.route('/api/benchmark/start', methods=['POST'])
def start_benchmark_api():
    """Triggers the benchmark script in a background thread."""
    data = request.json or {}
    n = int(data.get('n', 10000))
    k = int(data.get('k', 5))
    
    # Generate a unique ID for this run
    job_id = str(uuid.uuid4())
    BENCHMARK_JOBS[job_id] = {"status": "Running"}
    
    # Start the computation in a separate thread so Flask responds instantly
    thread = threading.Thread(target=run_benchmark_background, args=(job_id, n, k))
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/api/benchmark/status/<job_id>', methods=['GET'])
def check_benchmark_status(job_id):
    """Frontend calls this to check if computation is done."""
    job = BENCHMARK_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
        
    return jsonify(job)

# ==========================================
# FILE SERVING ROUTES
# ==========================================
@app.route('/audio/<path:filename>')
def serve_audio(filename):
    if not filename.endswith('.wav'):
        filename += '.wav'
    if not os.path.exists(os.path.join(AUDIO_FOLDER, filename)):
        return {"error": f"File {filename} not found"}, 404
    return send_from_directory(AUDIO_FOLDER, filename)

@app.route('/images/<path:filename>')
def serve_image(filename):
    if not os.path.exists(os.path.join(IMAGE_FOLDER, filename)):
        return {"error": "File not found"}, 404
    return send_from_directory(IMAGE_FOLDER, filename)

# --- Your original server-side R-Tree Plotly dump ---
@app.route('/visualize')
def serve_backend_visualization():
    media_type = request.args.get('type', 'image')
    # Grab the comma-separated list of IDs from the URL
    highlight_param = request.args.get('highlight', '') 
    
    # Convert "file1.wav,file2.wav" into a Python list
    highlight_list = [h.strip() for h in highlight_param.split(',')] if highlight_param else []
    
    if media_type == 'image':
        path = os.path.join(BASE_DIR, "data", "processed", "image_index.pkl")
    else:
        path = os.path.join(BASE_DIR, "data", "processed", "audio_index.pkl")
        
    try:
        # Pass the list to your new visualizer function
        html_graph = get_rtree_visualization(path, capacity=10, highlight_ids=highlight_list)
        return html_graph
    except Exception as e:
        return f"<h1>Visualization Error</h1><p>{str(e)}</p>", 500
    

if __name__ == '__main__':
    print("Starting Multi-Page R-Tree API server...")
    app.run(debug=True, use_reloader=False, port=5001)