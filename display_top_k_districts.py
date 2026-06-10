import json
import math
import matplotlib.pyplot as plt
import numpy as np
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

PRECINCT_FN = 'data/graph/grid_graph_10_by_10_2.json'
fname = 'local/output/grid10x10-2'

gdf = generate_shape_from_json(PRECINCT_FN)
fig, axes = plot_top_k_districts(fname + "/district_counts.json", gdf, top_k=20, cols=5, figsize=(20, 8))
plt.show()