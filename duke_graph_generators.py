import json


def generate_grid_adjacency_list(n, m):
    # Function to get the node index from grid coordinates
    def get_index(x, y):
        return x * m + y

    # Initialize adjacency list
    adjacency_list = {get_index(i, j): [] for i in range(n) for j in range(m)}
    node_info = {}
    # Populate adjacency list
    for i in range(n):
        for j in range(m):
            index = get_index(i, j)
            node_dict = {}
            node_dict['node_name'] = f"({i},{j})"
            node_dict['precinct_id_str'] = node_dict['node_name']
            node_dict['id'] = index

            if i > 0:  # Connect to the node above
                adjacency_list[index].append(get_index(i - 1, j))
            if i < n - 1:  # Connect to the node below
                adjacency_list[index].append(get_index(i + 1, j))
            if j > 0:  # Connect to the node on the left
                adjacency_list[index].append(get_index(i, j - 1))
            if j < m - 1:  # Connect to the node on the right
                adjacency_list[index].append(get_index(i, j + 1))

            node_dict['border_length'] = 4 - len(adjacency_list[index])
            node_dict['x_location'] = i
            node_dict['y_location'] = j
            node_dict['area'] = 1
            node_dict['population'] = 1

            node_info[index] = node_dict

    return adjacency_list, node_info


def generate_hexagonal_adjacency_list(rows, cols):
    def get_index(row, col):
        return row * cols + col

    adjacency_list = {get_index(row, col): [] for row in range(rows) for col in range(cols)}
    node_info = {}

    for row in range(rows):
        for col in range(cols):
            index = get_index(row, col)
            node_dict = {}
            node_dict['node_name'] = f"({row},{col})"
            node_dict['precinct_id_str'] = node_dict['node_name']
            node_dict['id'] = index
            # Even rows
            if row % 2 == 0:
                if col > 0:  # Left
                    adjacency_list[index].append(get_index(row, col - 1))
                if col < cols - 1:  # Right
                    adjacency_list[index].append(get_index(row, col + 1))
                if row > 0:  # Top left and top right
                    if col > 0:
                        adjacency_list[index].append(get_index(row - 1, col - 1))
                    adjacency_list[index].append(get_index(row - 1, col))
                if row < rows - 1:  # Bottom left and bottom right
                    if col > 0:
                        adjacency_list[index].append(get_index(row + 1, col - 1))
                    adjacency_list[index].append(get_index(row + 1, col))
            # Odd rows
            else:
                if col > 0:  # Left
                    adjacency_list[index].append(get_index(row, col - 1))
                if col < cols - 1:  # Right
                    adjacency_list[index].append(get_index(row, col + 1))
                if row > 0:  # Top left and top right
                    adjacency_list[index].append(get_index(row - 1, col))
                    if col < cols - 1:
                        adjacency_list[index].append(get_index(row - 1, col + 1))
                if row < rows - 1:  # Bottom left and bottom right
                    adjacency_list[index].append(get_index(row + 1, col))
                    if col < cols - 1:
                        adjacency_list[index].append(get_index(row + 1, col + 1))

            node_dict['border_length'] = 6 - len(adjacency_list[index])
            node_dict['x_location'] = col
            node_dict['y_location'] = row
            node_dict['area'] = 1
            node_dict['population'] = 1

            node_info[index] = node_dict

    return adjacency_list, node_info


def generate_triangular_adjacency_list(rows, cols):
    def get_index(row, col):
        return row * cols + col

    adjacency_list = {get_index(row, col): [] for row in range(rows) for col in range(cols)}
    node_info = {}

    for row in range(rows):
        for col in range(cols):
            index = get_index(row, col)
            node_dict = {}
            node_dict['node_name'] = f"({row},{col})"
            node_dict['precinct_id_str'] = node_dict['node_name'] # sigh
            node_dict['id'] = index
            if row > 0:  # Connect to the node above
                adjacency_list[index].append(get_index(row - 1, col))
                if col > 0:
                    adjacency_list[index].append(get_index(row - 1, col - 1))
                if col < cols - 1:
                    adjacency_list[index].append(get_index(row - 1, col + 1))
            if row < rows - 1:  # Connect to the node below
                adjacency_list[index].append(get_index(row + 1, col))
                if col > 0:
                    adjacency_list[index].append(get_index(row + 1, col - 1))
                if col < cols - 1:
                    adjacency_list[index].append(get_index(row + 1, col + 1))
            if col > 0:  # Connect to the node on the left
                adjacency_list[index].append(get_index(row, col - 1))
            if col < cols - 1:  # Connect to the node on the right
                adjacency_list[index].append(get_index(row, col + 1))

            node_dict['border_length'] = 8 - len(adjacency_list[index])
            node_dict['x_location'] = col
            node_dict['y_location'] = row
            node_dict['area'] = 1
            node_dict['population'] = 1

            node_info[index] = node_dict

    return adjacency_list, node_info


def adj_list_to_json(node_info, adj_list, outPath=None, districts=2):
    # this write the NetworkX/gerrychain json format out to disk.
    # This can also be read by CycleWalk.jl and the Forest RECOM code

    jsonDict = {"directed": False, "multigraph": False, "graph": []}
    jsonDict["nodes"] = []
    jsonDict["adjacency"] = []
    jsonDict['num_districts'] = districts

    for node in node_info:
        jsonDict['nodes'].append(node_info[node])
        edgeList = []
        for edge in adj_list[node]:
            edgeList.append({"id": edge, "length": 1})
        jsonDict["adjacency"].append(edgeList)

    if outPath:
        with open(outPath, 'w') as f:
            f.write(json.dumps(jsonDict, indent=4))
            return
    else:
        return jsonDict


if __name__ == '__main__':
    adj_list, node_info = generate_hexagonal_adjacency_list(100, 100)
    adj_list_to_json(node_info, adj_list, outPath='data/graph/hex_graph_100_by_100_2.json', districts=2)