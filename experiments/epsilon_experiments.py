import json
import gzip
import numpy as np
import matplotlib.pyplot as plt

def get_samples(jsonl_gz_path):
    all_samples = []

    with gzip.open(jsonl_gz_path, "rt") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line_idx < 3:
                continue

            plan = json.loads(line)

            # Parse districting and pair each district with its score
            districts = plan["districting"]

            all_samples.append(districts)
    return all_samples

def get_boundary_precincts(graph_data, districting):
    """
    Parameters
    ----------
    graph_data  : parsed JSON dict from the graph file
    districting : list of single-entry dicts, e.g. [{"[\"(1,1)\"]": 2}, ...]

    Returns
    -------
    set of node IDs that border a precinct in a different district
    """
    # Build coord string -> node_id lookup
    coord_to_id = {
        node["precinct_id_str"]: node["id"]
        for node in graph_data["nodes"]
    }

    # Flatten districting list and map to node_id -> district
    id_assignment = {}
    for entry in districting:
        for raw_key, district in entry.items():
            coord = json.loads(raw_key)[0]   # "[\"(1,1)\"]" -> "(1,1)"
            node_id = coord_to_id[coord]
            id_assignment[node_id] = district

    # Build simple adjacency dict: node_id -> [neighbour_id, ...]
    adjacency = {
        i: [nbr["id"] for nbr in neighbours]
        for i, neighbours in enumerate(graph_data["adjacency"])
    }

    # A precinct is on the boundary if any neighbour is in a different district
    return {
        node_id
        for node_id, neighbours in adjacency.items()
        if any(id_assignment[nbr] != id_assignment[node_id]
               for nbr in neighbours)
    }

# returns m and n, where m is number of rows, n is number of columns
# assumes graph is rectangular and begins at (0, 0)
def get_dimensions(graph_data):
    rightmost = max([node['x_location'] for node in graph_data['nodes']])
    topmost = max([node['y_location'] for node in graph_data['nodes']])
    return topmost + 1, rightmost + 1


def smallest_eps(m, n, boundary_precincts):
    leftmost = min([graph_data['nodes'][i]['x_location'] for i in boundary_precincts])
    rightmost = max([graph_data['nodes'][i]['x_location'] for i in boundary_precincts])

    mid = (n - 1) / 2
    x_dev = max(mid - leftmost, rightmost - mid)
    return x_dev / m

def plot_empirical_cdf(samples, ax=None, **plot_kwargs):
    """
    Plot the empirical CDF of a 1D array of float samples.

    Parameters
    ----------
    samples     : array-like of floats
    ax          : optional matplotlib Axes to plot onto
    **plot_kwargs: passed through to ax.step() (e.g. color, label, linewidth)
    """
    samples = np.sort(samples)
    n = len(samples)
    cdf = np.arange(1, n + 1) / n   # i/n for i = 1..n

    if ax is None:
        _, ax = plt.subplots()

    ax.step(samples, cdf, where="post", **plot_kwargs)
    ax.set_xlabel("Value")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)

    return ax


if __name__ == '__main__':
    fname = 'grid70x28-2'
    districtings = get_samples(f'../local/output/{fname}/atlas.jsonl.gz')
    epsilons = []
    with open('../data/graph/grid_graph_70_by_28_2.json', 'r') as f:
        graph_data = json.load(f)
    m, n = get_dimensions(graph_data)

    for districting in districtings:
        boundary_precincts = get_boundary_precincts(graph_data, districting)
        epsilon = smallest_eps(m, n, boundary_precincts)
        epsilons.append(epsilon)

    fig, ax = plt.subplots()
    plot_empirical_cdf(epsilons, ax=ax, color="steelblue", label="Empirical CDF")
    ax.legend()
    plt.show()