import networkx as nx
import json
from gerrychain.tree import recursive_tree_part
from gerrychain import Graph

# Code courtesy of Gemini 3.1 Pro

# 1. Load your graph (assuming it already has population data)
# G = ...

name = 'data/networkx/grid_graph_5_by_5_5'

fname = f'{name}.json'
outname = f'{name}_seeded.json'

with open(fname, "r") as f:
    data = json.load(f)

# G = nx.json_graph.adjacency_graph(data)
G = Graph.from_json(fname)

num_districts = 5
total_pop = sum(G.nodes[n]["population"] for n in G.nodes)
target_pop = total_pop / num_districts

# 2. Generate a random seed plan
# This partitions the graph into 10 contiguous, population-balanced districts
seed_assignment = recursive_tree_part(
    G,
    parts=range(1, num_districts+1),
    pop_target=target_pop,
    pop_col="population",
    epsilon=0.1
)

# 3. Add the generated assignment to your graph nodes
nx.set_node_attributes(G, seed_assignment, "SEED_PLAN")

# 4. Export to the JSON format frcw.rs expects
data = nx.readwrite.json_graph.adjacency_data(G)
with open(outname, "w") as f:
    json.dump(data, f)