import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from data import generate_shape_from_json
from utils import plot_district

def plot_top_k_districts(json_path, gdf, top_k=10, cols=5, figsize=(20, 8)):
    # Build lookup: str((0,0)) -> integer row index in gdf
    precinct_to_idx = {str(p): i for i, p in enumerate(gdf["precinct"])}

    with open(json_path) as f:
        data = json.load(f)

    # Sort by count descending and take top_k
    data_sorted = sorted(data, key=lambda x: x["count"], reverse=True)[:top_k]

    rows = math.ceil(top_k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for idx, entry in enumerate(data_sorted):
        # Parse each precinct string e.g. "[\"(2,0)\"]" -> "(2,0)" -> int index
        precinct_indices = [
            precinct_to_idx[json.loads(p)[0]]
            for p in entry["precincts"]
        ]
        plot_district(gdf, precinct_indices, ax=axes_flat[idx])
        axes_flat[idx].set_title(
            f"#{idx+1}  count={entry['count']:,}\niso={entry['isoperimetric_score']:.2f}",
            fontsize=8,
        )

    for ax in axes_flat[top_k:]:
        ax.axis("off")

    fig.suptitle(f"Top {top_k} Most Frequent Districts", fontsize=12)
    return fig, axes

def plot_k_districts(json_path, gdf, k=10, start=1, cols=5, figsize=(20, 8)):
    """
    Plot the most frequent districts from a JSON file.

    Parameters
    ----------
    json_path : path to the JSON file
    gdf       : GeoDataFrame with a 'precinct' column
    top_k     : number of districts to display (default 10)
    start     : rank to start from, 1-indexed (default 1)
    cols      : number of columns in the grid (default 5)
    figsize   : figure size (default (20, 8))
    """
    precinct_to_idx = {str(p): i for i, p in enumerate(gdf["precinct"])}

    with open(json_path) as f:
        data = json.load(f)

    data_sorted = sorted(data, key=lambda x: x["count"], reverse=True)

    if start < 1:
        raise ValueError("start must be >= 1")

    # Clip to available range
    start_idx = start - 1
    end_idx = min(start_idx + k, len(data_sorted))

    if start_idx >= len(data_sorted):
        raise ValueError(f"start={start} exceeds the number of available entries ({len(data_sorted)})")

    selected = data_sorted[start_idx:end_idx]
    n_displayed = len(selected)

    rows = math.ceil(n_displayed / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for idx, entry in enumerate(selected):
        rank = start_idx + idx + 1  # 1-indexed rank for title
        precinct_indices = [
            precinct_to_idx[json.loads(p)[0]]
            for p in entry["precincts"]
        ]
        plot_district(gdf, precinct_indices, ax=axes_flat[idx])
        axes_flat[idx].set_title(
            f"#{rank}  count={entry['count']:,}\niso={entry['isoperimetric_score']:.2f}",
            fontsize=8,
        )

    for ax in axes_flat[n_displayed:]:
        ax.axis("off")

    fig.suptitle(f"#{start}–#{start_idx + n_displayed} Most Frequent Districts", fontsize=12)
    return fig, axes

def save_top_k_districts(json_path, gdf, output_dir=None, k=20, cols=5, figsize=(20, 8), folder='out/'):
    """
    Save images of the most frequent districts in batches of k into output_dir.
    Files are named e.g. districts_1_20.png, districts_21_40.png, etc.

    Parameters
    ----------
    json_path  : path to the JSON file
    gdf        : GeoDataFrame with a 'precinct' column
    output_dir : directory to save images into (created if it doesn't exist)
    k          : batch size / number of districts per image (default 20)
    cols       : number of columns in the grid (default 5)
    figsize    : figure size (default (20, 8))
    """
    if output_dir is None:
        output_dir = f'local/district_count_graphs/{folder}'

    os.makedirs(output_dir, exist_ok=True)

    precinct_to_idx = {str(p): i for i, p in enumerate(gdf["precinct"])}

    with open(json_path) as f:
        data = json.load(f)

    data_sorted = sorted(data, key=lambda x: x["count"], reverse=True)
    total = len(data_sorted)

    for batch_start in range(0, total, k):
        batch_end = min(batch_start + k, total)
        selected = data_sorted[batch_start:batch_end]
        n_displayed = len(selected)

        rows = math.ceil(n_displayed / cols)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
        axes_flat = np.array(axes).flatten()

        for idx, entry in enumerate(selected):
            rank = batch_start + idx + 1
            precinct_indices = [
                precinct_to_idx[json.loads(p)[0]]
                for p in entry["precincts"]
            ]
            plot_district(gdf, precinct_indices, ax=axes_flat[idx])
            axes_flat[idx].set_title(
                f"#{rank}  count={entry['count']:,}\niso={entry['isoperimetric_score']:.2f}",
                fontsize=8,
            )

        for ax in axes_flat[n_displayed:]:
            ax.axis("off")

        start_rank = batch_start + 1
        end_rank = batch_start + n_displayed
        fig.suptitle(f"#{start_rank}–#{end_rank} Most Frequent Districts", fontsize=12)

        fname = os.path.join(output_dir, f"districts_{start_rank}_{end_rank}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


PRECINCT_FN = 'data/graph/grid_graph_10_by_10_2.json'
fname = 'local/output/grid10x10-2'

gdf = generate_shape_from_json(PRECINCT_FN)
# fig, axes = plot_top_k_districts(fname + "/district_counts.json", gdf, top_k=20, cols=5, figsize=(20, 8))
save_top_k_districts(fname + '/district_counts.json', gdf, k=20, folder='grid10x10-2-1e5')
plt.show()