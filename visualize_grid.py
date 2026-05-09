import os
import json
import matplotlib.pyplot as plt

"""
Visualizes given grid graph using matplotlib. Plots adjacencies as lines,
as well as precincts, i.e. nodes, as dots based on their location. 

Currently requires manually going into the file and changing the input
filepath. Runs with the standard run button.

@author Edward Zhang
"""

# TODO: Change to command line interface
# TODO: add toggle to save image instead of, or jointly with displaying

fname = os.getcwd() + '/data/graph/grid_graph_5_by_5.json'

with open(fname, 'r') as f:
    data = json.load(f)

locations = dict()

for node in data['nodes']:
    locations[node['precinct_id']] = (node['x_location'], node['y_location'])

fig, ax = plt.subplots()

for node_id, neighbors in enumerate(data["adjacency"]):
    x0, y0 = locations[node_id]
    for neighbor in neighbors:
        x1, y1 = locations[neighbor["id"]]
        ax.plot([x0, x1], [y0, y1], color='steelblue')

for node_id, (x, y) in locations.items():
    ax.scatter(x, y, color='tomato', zorder=2)
    ax.text(x + 0.1, y + 0.05, str(node_id), ha="center", va="bottom", fontsize=7)

ax.set_title("Grid visualization")
ax.set_xlabel("x")
ax.set_ylabel("y")
x_vals = [x for x, y in locations.values()]
y_vals = [y for x, y in locations.values()]
ax.set_xticks(range(min(x_vals), max(x_vals) + 1))
ax.set_yticks(range(min(y_vals), max(y_vals) + 1))
ax.set_aspect("equal")
ax.grid(False)
plt.tight_layout()
plt.show()