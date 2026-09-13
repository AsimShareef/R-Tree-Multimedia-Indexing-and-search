# DBMS Term Project: High-Dimensional Multimedia Retrieval

This project implements a multimedia similarity-search system for images and audio using high-dimensional feature vectors.

Core ideas:
- Feature extraction from raw media
- PCA-based dimensionality reduction
- Custom R-Tree indexing for KNN retrieval
- Custom KD-Tree and linear scan for benchmark comparison
- Flask web app for search, visualization, and benchmarking

## Project Structure

- `build_index.py`: Builds database indexes and PCA models from raw media
- `app.py`: Flask web server (UI + APIs)
- `src/index/`: R-Tree and KD-Tree implementations
- `src/search/`: Query execution and KNN search logic
- `src/evaluation/`: Visualizer + benchmark logic
- `templates/`: Frontend pages
- `data/raw_images`, `data/raw_audio`: Input datasets
- `data/processed`: Generated PCA models and tree indexes

## Requirements

- Python 3.9+
- pip
- Optional: GNU Make (for running Makefile commands)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Raw media is not committed to this repo (it's large and not ours to redistribute). Populate these folders yourself before building the index:

- `data/raw_images/` — any collection of `.jpg`/`.png` images
- `data/raw_audio/` — `.wav` files; developed and tested against [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)

`data/processed/` ships with a small pre-built PCA model + R-Tree index so the search/visualizer/benchmark pages work out of the box without rebuilding.

## Quick Start

1. Put dataset files in `data/raw_images` and `data/raw_audio` (see [Dataset](#dataset)).

2. Build indexes and PCA models:

```bash
python build_index.py
```

3. Start the web app:

```bash
python app.py
```

4. Open browser:

```text
http://127.0.0.1:5001
```

## Benchmarking

There are two benchmark paths:

1. Web benchmark page
- Start `app.py`
- Open `/benchmark`
- Uses background job APIs:
	- `POST /api/benchmark/start`
	- `GET /api/benchmark/status/<job_id>`

2. Direct benchmark script

```bash
python src/evaluation/benchmark.py
```

This compares linear scan, custom KD-Tree, and custom R-Tree across dimensions.

## Common Workflow

1. Add/update media in raw data folders
2. Rebuild indexes with `build_index.py`
3. Run `app.py`
4. Test search and visualizer pages
5. Run benchmark page or script for performance analysis

## Troubleshooting

- `ModuleNotFoundError`:
	- Run commands from project root.
- Empty search results or errors:
	- Ensure `data/processed` has generated files after running `build_index.py`.
- Missing media in UI:
	- Confirm query/output filenames match files in `data/raw_images` and `data/raw_audio`.

## Notes

- Benchmark runtime increases with larger N and higher dimensions.
- Initial feature extraction (especially image/audio models) may take time.
