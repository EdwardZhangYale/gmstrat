import networkx as nx
import json
import os
import argparse

# Courtesy of Gemini 3.1 Pro

def generate_l_shaped_grid(m: int, n: int, cw: int, ch: int, num_districts: int, output_file: str):
    """
    Generates a JSON representing the dual graph of an m x n grid
    with a cw x ch rectangular corner removed from the top-right.
    """
    # Sanity check to ensure we don't delete the whole graph
    if cw >= n or ch >= m:
        raise ValueError("Corner dimensions must be strictly smaller than grid dimensions.")

    # 1. Generate the base grid graph
    G = nx.grid_2d_graph(n, m)

    # 2. Identify and remove the corner nodes (Top-Right corner in this case)
    # n is width (x), m is height (y)
    nodes_to_remove = [
        (x, y)
        for x in range(n - cw, n)
        for y in range(m - ch, m)
    ]
    G.remove_nodes_from(nodes_to_remove)

    # 3. Add graph-level attributes
    G.graph["num_districts"] = num_districts

    # 4. Add node and edge attributes
    # Using enumerate ensures IDs are contiguous (0 to N-1), which frcw.rs needs
    for idx, node in enumerate(G.nodes()):
        x, y = node

        # The math `4 - degree` works perfectly here too!
        # NetworkX automatically updated the degrees when we removed the corner,
        # so the new "inner" corner boundaries are perfectly accounted for.
        border_length = 4 - G.degree[node]

        G.nodes[node].update({
            "id": idx,
            "precinct_id": idx,
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

    # 5. Relabel nodes from (x, y) tuples to our new contiguous integer IDs
    mapping = {node: G.nodes[node]["id"] for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    # 6. Export using NetworkX's built-in adjacency formatter
    data = nx.readwrite.json_graph.adjacency_data(G)

    data['num_districts'] = num_districts

    # 7. Save to disk
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Successfully generated L-shaped grid at {output_file}")
    print(f"Total nodes remaining: {G.number_of_nodes()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an L-shaped grid graph JSON.")
    parser.add_argument("-m", type=int, default=100, help="Grid height")
    parser.add_argument("-n", type=int, default=100, help="Grid width")
    parser.add_argument("--cw", type=int, default=20, help="Width of the corner to remove")
    parser.add_argument("--ch", type=int, default=20, help="Height of the corner to remove")
    parser.add_argument("-l", "--districts", type=int, default=6, help="Number of districts")

    args = parser.parse_args()

    fname = os.path.join(
        os.getcwd(),
        f"data/networkx/l_grid_{args.m}x{args.n}_minus_{args.ch}x{args.cw}_{args.districts}.json"
    )

    generate_l_shaped_grid(args.m, args.n, args.cw, args.ch, args.districts, fname)