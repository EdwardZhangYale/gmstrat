import networkx as nx
import json
import os
import argparse

# Code courtesy of Gemini 3.1 Pro

def generate_grid_graph(m: int, n: int, num_districts: int, output_file: str):
    """
    Generates a JSON representing the dual graph of precincts in an m x n grid.
    """
    # 1. Generate the base grid graph using NetworkX
    # By default, nodes are tuples like (0,0), (0,1), etc.
    G = nx.grid_2d_graph(n, m)

    # 2. Add graph-level attributes (must be in G.graph to export correctly)
    G.graph["num_districts"] = num_districts

    # 3. Add node and edge attributes
    for node in G.nodes():
        x, y = node

        # Calculate node ID to match your original math
        node_id = x * m + y

        # In a grid, the number of boundaries a cell touches is 4 minus its degree
        # (e.g., a corner has degree 2, so it touches 2 borders)
        border_length = 4 - G.degree[node]

        # Assign node attributes
        G.nodes[node].update({
            "id": node_id,
            "precinct_id": node_id,
            "precinct_id_str": f"({x},{y})",
            "border_length": border_length,
            "x_location": x,
            "y_location": y,
            "area": 1,
            "population": 1,
            "county": "A",
        })

    # Add default lengths to edges
    for u, v in G.edges():
        G.edges[u, v]["length"] = 1

    # 4. Relabel nodes from (x, y) tuples to integer IDs
    # NetworkX json exporter prefers scalar IDs rather than tuples
    mapping = {node: G.nodes[node]["id"] for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    # 5. Export using NetworkX's built-in adjacency formatter
    data = nx.readwrite.json_graph.adjacency_data(G)

    # 6. Save to disk
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Successfully generated grid graph JSON at {output_file}")


if __name__ == "__main__":
    # Added a simple CLI so you don't have to hardcode variables!
    parser = argparse.ArgumentParser(description="Generate a grid graph JSON.")
    parser.add_argument("-m", type=int, default=100, help="Grid height")
    parser.add_argument("-n", type=int, default=100, help="Grid width")
    parser.add_argument("-l", "--districts", type=int, default=3, help="Number of districts")

    args = parser.parse_args()

    # Construct filename
    fname = os.path.join(os.getcwd(), f"data/networkx/grid_graph_{args.m}_by_{args.n}_{args.districts}.json")

    generate_grid_graph(args.m, args.n, args.districts, fname)