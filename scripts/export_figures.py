#!/usr/bin/env python3
"""
Export PNG figures from Jupyter notebooks.

Scans notebooks for specific cells by keyword matching in the source code,
extracts base64-encoded PNG image data from cell outputs, and saves them
as PNG files to docs/figures/.

Usage:
    python scripts/export_figures.py
"""

import json
import base64
import os
import sys

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")

# Figure extraction specifications.
# Each entry maps (notebook_filename, identifying_keywords, output_filename).
# Keywords are checked against the cell source code (all must match).
# The first cell matching all keywords AND having an image/png output is used.
FIGURE_SPECS = [
    # --- NB 04: Behavior Pipeline ---
    {
        "notebook": "04_behavior_pipeline.ipynb",
        "output": "fig-video-frames.png",
        "keywords": ["subplots(1, 2", "solo_frame", "img_solo"],
        "description": "Raw video frames (solo vs social)",
    },
    {
        "notebook": "04_behavior_pipeline.ipynb",
        "output": "fig-sleap-skeleton.png",
        "keywords": ["skeleton", "frame_to_lf"],
        "description": "SLEAP skeleton overlay on video frames",
    },
    {
        "notebook": "04_behavior_pipeline.ipynb",
        "output": "fig-distance-threshold.png",
        "keywords": ["THRESH", "distance"],
        "description": "Inter-animal distance with threshold",
    },
    {
        "notebook": "04_behavior_pipeline.ipynb",
        "output": "fig-social-comparison.png",
        "keywords": ["social", "not social", "looks like"],
        "description": "Social vs not-social video frame comparison",
    },
    # --- NB 03: Spatial Clusters ---
    {
        "notebook": "03_spatial_clusters.ipynb",
        "output": "fig-correlation-image.png",
        "keywords": ["footprint", "composite", "Cn"],
        "description": "Correlation image with cluster footprint overlays",
    },
    {
        "notebook": "03_spatial_clusters.ipynb",
        "output": "fig-spatial-scatter.png",
        "keywords": ["Centroid scatter", "centroids_yx"],
        "description": "Neuron centroid scatter colored by cluster",
    },
    {
        "notebook": "03_spatial_clusters.ipynb",
        "output": "fig-cluster-rasters.png",
        "keywords": ["raster", "cal_z_sorted", "Cluster boundary"],
        "description": "Full-session activity raster with cluster boundary and social shading",
    },
    {
        "notebook": "03_spatial_clusters.ipynb",
        "output": "fig-intensity-maps.png",
        "keywords": ["delta", "theta", "dt_power"],
        "description": "Delta+theta fractional power intensity maps",
    },
    # --- NB 01: EDA ---
    {
        "notebook": "01_eda.ipynb",
        "output": "fig-neuron-traces.png",
        "keywords": ["top_neurons", "n_show", "shade_social"],
        "description": "Top active neuron traces with social epoch shading",
    },
    {
        "notebook": "01_eda.ipynb",
        "output": "fig-band-decomposition.png",
        "keywords": ["Band-filtered", "bandpass_filter", "FREQ_BANDS"],
        "description": "Band-decomposed filtered signals with behavior overlay",
    },
    {
        "notebook": "01_eda.ipynb",
        "output": "fig-smi-modulation.png",
        "keywords": ["SMI", "smi_df", "hist"],
        "description": "SMI distribution across all neurons",
    },
    # --- NB 02: Band-Neuron-Behavior ---
    {
        "notebook": "02_band_neuron_behavior.ipynb",
        "output": "fig-q1-band-power.png",
        "keywords": ["effect size", "Cohen"],
        "description": "Band power comparison with Cohen's d effect sizes",
    },
    {
        "notebook": "02_band_neuron_behavior.ipynb",
        "output": "fig-q2-clusters.png",
        "keywords": ["cluster", "centroid", "heatmap"],
        "description": "Neuron cluster centroid profiles and heatmap",
    },
    {
        "notebook": "02_band_neuron_behavior.ipynb",
        "output": "fig-q3-classification.png",
        "keywords": ["AUC", "clf_results"],
        "description": "Classification AUC comparison across strategies",
    },
]


def load_notebook(path):
    """Load a Jupyter notebook as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cell_source(cell):
    """Get the full source code of a cell as a single string."""
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    return source


def cell_has_png(cell):
    """Check if a cell has at least one image/png output."""
    for output in cell.get("outputs", []):
        if "data" in output and "image/png" in output["data"]:
            return True
    return False


def extract_png_data(cell):
    """Extract the first base64 PNG data from a cell's outputs.

    Returns the raw bytes of the PNG, or None if not found.
    """
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "image/png" in data:
            png_b64 = data["image/png"]
            # Handle both string and list formats
            if isinstance(png_b64, list):
                png_b64 = "".join(png_b64)
            # Strip whitespace/newlines that may be in the base64
            png_b64 = png_b64.strip()
            return base64.b64decode(png_b64)
    return None


def find_cell_by_keywords(notebook, keywords):
    """Find the first code cell whose source contains ALL keywords and has a PNG output.

    Returns (cell_index, cell) or (None, None) if not found.
    """
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        if not cell_has_png(cell):
            continue
        source = get_cell_source(cell).lower()
        if all(kw.lower() in source for kw in keywords):
            return i, cell
    return None, None


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Cache loaded notebooks to avoid re-reading
    nb_cache = {}

    success_count = 0
    fail_count = 0

    print(f"Exporting figures to: {FIGURES_DIR}")
    print(f"{'='*70}")

    for spec in FIGURE_SPECS:
        nb_name = spec["notebook"]
        output_name = spec["output"]
        keywords = spec["keywords"]
        desc = spec["description"]

        # Load notebook (cached)
        if nb_name not in nb_cache:
            nb_path = os.path.join(NOTEBOOKS_DIR, nb_name)
            if not os.path.exists(nb_path):
                print(f"  SKIP  {output_name} -- notebook not found: {nb_name}")
                fail_count += 1
                continue
            nb_cache[nb_name] = load_notebook(nb_path)

        nb = nb_cache[nb_name]

        # Find the matching cell
        cell_idx, cell = find_cell_by_keywords(nb, keywords)

        if cell_idx is None:
            print(f"  FAIL  {output_name} -- no cell matched keywords {keywords} in {nb_name}")
            fail_count += 1
            continue

        # Extract PNG data
        png_bytes = extract_png_data(cell)
        if png_bytes is None:
            print(f"  FAIL  {output_name} -- cell {cell_idx} has no extractable PNG data")
            fail_count += 1
            continue

        # Save to file
        out_path = os.path.join(FIGURES_DIR, output_name)
        with open(out_path, "wb") as f:
            f.write(png_bytes)

        size_kb = len(png_bytes) / 1024
        print(f"  OK    {output_name:30s} <- {nb_name} cell {cell_idx:>3d}  ({size_kb:7.1f} KB)  {desc}")
        success_count += 1

    print(f"{'='*70}")
    print(f"Done: {success_count} exported, {fail_count} failed, {len(FIGURE_SPECS)} total")

    if fail_count > 0:
        print("\nFailed figures may need manual keyword adjustment.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
