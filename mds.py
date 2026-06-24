import pandas as pd
from sklearn.manifold import MDS
from utils import str_to_vec
import json
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform


def compute_hamming_distance_matrix(json_path, output_path=None, k=None):
    """
    Compute the pairwise Hamming distance matrix between districts in a JSON
    file and save it as a .npy file in condensed form (upper triangle).

    Parameters
    ----------
    json_path   : path to the input JSON file
    output_path : path to save the .npy file (e.g. "hamming.npy")
    k           : if specified, only compute distances for the top k most
                  frequent districts (by count). If None, use all districts.
    """
    with open(json_path) as f:
        data = json.load(f)

    data_sorted = sorted(data, key=lambda x: x["count"], reverse=True)
    if k is not None:
        data_sorted = data_sorted[:k]

    precinct_sets = [
        frozenset(json.loads(p)[0] for p in entry["precincts"])
        for entry in data_sorted
    ]

    all_precincts = sorted(set(p for s in precinct_sets for p in s))
    precinct_to_idx = {p: i for i, p in enumerate(all_precincts)}

    n = len(precinct_sets)
    m = len(all_precincts)

    X = np.zeros((n, m), dtype=np.uint8)
    for i, s in enumerate(precinct_sets):
        for p in s:
            X[i, precinct_to_idx[p]] = 1

    condensed = pdist(X, metric='cityblock')

    if output_path is not None:
        np.save(output_path, condensed)
        print(f"Saved condensed Cityblock distance matrix ({n} districts) to {output_path}")
    return condensed

# D = np.load(f"local/output/grid{N}x{N}_k{k}/processed/distance_matrix.npy").astype(float)
N = 10

json_path = 'local/output/grid10x10-2/district_counts.json'

D = squareform(compute_hamming_distance_matrix(json_path, k=10000)).astype(float)
k = len(D)

mds = MDS(n_components=2, metric="precomputed", init="random", random_state=42, normalized_stress="auto")
coords = mds.fit_transform(D)   # shape (130, 2)

with open(json_path) as f:
    data = json.load(f)

data_sorted = sorted(data, key=lambda x: x["count"], reverse=True)[:k]
total_counts = sum(x["count"] for x in data_sorted)

uid_to_str  = {i: json.dumps(entry["precincts"]) for i, entry in enumerate(data_sorted)}
uid_to_freq = {i: entry["count"] / total_counts   for i, entry in enumerate(data_sorted)}
# total number of times any district was sampled
# total_counts = df_districts["count"].sum()

# df_sorted = df_districts.copy()
# df_sorted["freq"] = df_sorted["count"] / total_counts
# df_sorted = df_sorted.sort_values("freq", ascending=False).reset_index(drop=True)

# uid_to_str  = df_districts.set_index("district_uid")["district_str"].to_dict()
# uid_to_freq = df_sorted.set_index("district_uid")["freq"].to_dict()

# ratios = np.array([
#     isoperimetric_ratio(str_to_vec(uid_to_str[uid]).tolist(), N)
#     for uid in range(len(coords))
# ])

ratios = np.array([
    np.sqrt(data_sorted[uid]["isoperimetric_score"])
    for uid in range(len(coords))
])

freqs = np.array([uid_to_freq[uid] for uid in range(len(coords))])

top4_idx = np.argsort(freqs)[-4:]

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(coords[:, 0], coords[:, 1], c=ratios, cmap="plasma",
                s=60, linewidths=0.4, edgecolors="grey", zorder=2)
plt.colorbar(sc, ax=ax, label="Isoperimetric ratio P/√A")

ax.scatter(coords[top4_idx, 0], coords[top4_idx, 1],
           s=200, facecolors="none", edgecolors="red", linewidths=1.8, zorder=3,
           label="Top 4 by frequency")
ax.legend(loc="best", fontsize=10)

ax.set_title(
    f"MDS embedding of district distance matrix — {N}×{N} grid, k={k}\n"
    f"(colour = P/√A, red circles = top 4 marginal frequency)",
    fontsize=12,
)
ax.set_xlabel("MDS dim 1")
ax.set_ylabel("MDS dim 2")
plt.tight_layout()
plt.show()

print(f"MDS stress: {mds.stress_:.4f}")
print("Top 4 districts by frequency:")
for uid in top4_idx[np.argsort(freqs[top4_idx])[::-1]]:
    print(f"  uid={uid}  freq={freqs[uid]:.4f}  P/√A={ratios[uid]:.2f}  {uid_to_str[uid]}")