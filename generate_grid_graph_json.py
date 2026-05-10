import json
import os

m = 3
n = 3
l = 3

def node_id(x, y):
    return x * m + y

def border_length(x, y):
    # Number of grid boundary edges touching this cell
    return (x == 0) + (x == n - 1) + (y == 0) + (y == m - 1)

nodes = []
for x in range(n):
    for y in range(m):
        nodes.append({
            "precinct_id": node_id(x, y),
            "id": node_id(x, y),
            "precinct_id_str": f"({x},{y})",
            "border_length": border_length(x, y),
            "x_location": x,
            "y_location": y,
            "area": 1,
            "population": 1,
            "county": "A",
        })

adjacency = [[] for _ in range(m * n)]
for x in range(n):
    for y in range(m):
        nid = node_id(x, y)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < n and 0 <= ny_ < m:
                adjacency[nid].append({"id": node_id(nx_, ny_), "length": 1})

graph = {
    "directed": False,
    "multigraph": False,
    "graph": [],
    "num_districts": l,
    "nodes": nodes,
    "adjacency": adjacency,
}

fname = os.getcwd() + f'/data/graph/grid_graph_{m}_by_{n}.json'

with open(fname, "w") as f:
    json.dump(graph, f, indent=4)