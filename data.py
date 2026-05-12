import json

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

# Wonderful annotations by Claude Sonnet 4.6

# Generates a JSON graph representation of an N×N grid, where each cell is a
# node and edges connect horizontally/vertically adjacent cells. The output
# file is formatted for use with redistricting tools (e.g. CycleWalk) that
# expect a graph JSON with node attributes and an adjacency list.
#
# Parameters:
#   N                  : grid side length (produces an N×N grid of N² nodes)
#   filename           : path to write the JSON file to
#   population_matrix  : optional N×N array of population counts per cell;
#                        if None, every cell is assigned population 1
#   num_districts      : (keyword-only) number of districts to partition into
def generate_grid_graph(N, filename, population_matrix=None, *, num_districts: int):
    nodes = []
    adjacency = [[] for _ in range(N * N)]  # one adjacency list per node

    # Convert the optional population matrix to a numpy array for easy indexing
    pop = None
    if population_matrix is not None:
        pop = np.asarray(population_matrix)

    # Helper: map 2D grid coordinates (x, y) to a flat node index.
    # Nodes are indexed row-major: node (x, y) → index x*N + y.
    def node_id(x, y):
        return x * N + y

    for x in range(N):
        for y in range(N):
            idx = node_id(x, y)

            # Count how many sides of this cell lie on the outer boundary of
            # the grid. Corner cells contribute 2, edge cells 1, interior 0.
            border_length = 0
            if x == 0:
                border_length += 1
            if x == N - 1:
                border_length += 1
            if y == 0:
                border_length += 1
            if y == N - 1:
                border_length += 1

            # Build the node attribute dictionary expected by the graph format
            node = {
                "id": idx,
                "precinct_id": idx,
                "precinct_id_str": f"({x},{y})",   # human-readable coordinate label
                "border_length": border_length,
                "x_location": x,
                "y_location": y,
                "area": 1,                          # unit area for each cell
                "population": int(pop[x, y]) if pop is not None else 1,
                "county": "A",                      # single county — no county structure
            }
            nodes.append(node)

            # Add edges to each of the (up to 4) orthogonal neighbors.
            # Each edge has length 1, representing one shared cell boundary.
            # Only add edges in the "forward" directions (right and up) to
            # avoid adding each edge twice? Actually no — both directions are
            # added, giving a symmetric undirected adjacency list.
            if x > 0:
                adjacency[idx].append({"id": node_id(x - 1, y), "length": 1})  # left
            if x < N - 1:
                adjacency[idx].append({"id": node_id(x + 1, y), "length": 1})  # right
            if y > 0:
                adjacency[idx].append({"id": node_id(x, y - 1), "length": 1})  # down
            if y < N - 1:
                adjacency[idx].append({"id": node_id(x, y + 1), "length": 1})  # up

    # Assemble the full graph JSON object
    graph_json = {
        "directed": False,
        "multigraph": False,
        "graph": [],
        "num_districts": int(num_districts),
        "nodes": nodes,
        "adjacency": adjacency,
    }

    # Write to file and also return the dict for in-memory use
    with open(filename, "w") as f:
        json.dump(graph_json, f, indent=4)
    return graph_json


# Generates a GeoDataFrame of square polygons representing an N×N grid,
# suitable for use with geopandas for plotting or spatial joins. Each cell
# becomes a Shapely Polygon with a given side length (cell_size).
#
# Parameters:
#   N         : grid side length (produces N² cells)
#   cell_size : side length of each square cell in the projected CRS units
#               (default 1.0, meaning 1 metre if using EPSG:3857)
#   crs       : coordinate reference system for the GeoDataFrame
#               (default EPSG:3857, a metres-based Web Mercator projection)
def generate_grid_shape(N, cell_size=1.0, *, crs="EPSG:3857"):
    polys = []
    ids = []
    precincts = []

    for i in range(N):       # i indexes rows (y direction)
        for j in range(N):   # j indexes columns (x direction)

            # Compute the bottom-left and top-right corners of this cell.
            # j controls the horizontal (x) position, i controls the vertical (y).
            x0, y0 = j * cell_size, i * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size

            # Create the cell as a Shapely Polygon (counter-clockwise corners)
            poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            polys.append(poly)

            # Flat index and string label for this cell, consistent with the
            # node indexing used in generate_grid_graph
            ids.append(i * N + j)
            precincts.append(f"({i},{j})")

    return gpd.GeoDataFrame({"id": ids, "precinct": precincts}, geometry=polys, crs=crs)