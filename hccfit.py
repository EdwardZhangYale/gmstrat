import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import random
import time
from tqdm import tqdm

# Annotations by Claude Sonnet 4.6

class HccLinkage:
    # Implements Hierarchical Complete-linkage Clustering via an ultrametric,
    # using a voting-based criterion to decide when two clusters should merge.
    #
    # The key idea: rather than merging clusters based on minimum/maximum/average
    # pairwise distance (as in standard agglomerative clustering), this algorithm
    # processes edges in distance order and merges two clusters only when
    # *every node in each cluster has seen a majority of the other cluster*.
    # This majority-vote condition produces an ultrametric — a hierarchy where
    # the merge height between any two points equals the distance at which their
    # clusters satisfied the mutual majority condition.
    #
    # Parameters:
    #   d    : (n, n) symmetric pairwise distance matrix
    #   alt  : unused flag (reserved for an alternate algorithm variant)
    #   rand : unused flag (reserved for randomized edge ordering)
    #   tol  : tolerance for floating point comparisons (currently unused)
    def __init__(self, d, alt=False, rand=False, tol=1e-5):
        self.d = d
        self.n = self.d.shape[0]          # number of leaf nodes (districts)
        self.alt = alt
        self.tol = tol

        # Stack of available internal node IDs, assigned to new nodes as clusters
        # merge. Internal nodes are numbered n, n+1, ..., 2n-1 (reversed so
        # .pop(-1) gives the next ID in ascending order: n, n+1, n+2, ...)
        self.nextroots = list(range(self.n, 2 * self.n))
        self.nextroots.reverse()

        # The ultrametric distance matrix: d_U[i,j] is set to the merge height
        # of nodes i and j when their clusters merge. Starts as zeros.
        self.d_U = np.zeros((self.n, self.n))

        # A[i,j] = number of edges processed so far between leaf i and leaf j.
        # Tracks which pairs have been "seen" by the algorithm.
        self.A = np.zeros((self.n, self.n))

        # N[v, c] = number of edges processed between leaf v and leaves currently
        # in cluster c. Used to measure how much of cluster c leaf v has "seen".
        self.N = np.zeros((self.n, 2 * self.n))

        # H[v, c] = 1 if leaf v has seen a majority of cluster c (i.e.
        # 2 * N[v, c] >= S[c]). This is the per-leaf majority flag.
        self.H = np.zeros((self.n, 2 * self.n))

        # M[k, c] = number of leaves in cluster k that have seen a majority of
        # cluster c (i.e. sum of H[v, c] for all v in cluster k).
        # The merge condition is: M[k,l] + M[l,k] == S[k] + S[l], meaning
        # every leaf in each cluster has seen a majority of the other cluster.
        self.M = np.zeros((2 * self.n, 2 * self.n))

        # S[c] = size (number of leaf members) of cluster c.
        # Leaf nodes start with size 1; internal nodes accumulate sizes on merge.
        self.S = np.ones(2 * self.n)

        # membership[v] = current cluster ID of leaf v.
        # Starts as v (each leaf in its own cluster), updated as clusters merge.
        self.membership = np.arange(self.n)

        # NetworkX graph representing the dendrogram tree.
        # Nodes are cluster IDs; edges connect child clusters to their parent,
        # weighted by the difference in heights (branch lengths).
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))

        # heights[c] = the distance value at which cluster c was created by a merge.
        # Leaf heights are 0 (they exist from the start).
        self.heights = np.zeros(2 * self.n)

        # Z is the standard scipy-format linkage matrix, shape (n-1, 4).
        # Each row: [left_child, right_child, merge_distance, cluster_size]
        # Row index r corresponds to internal node (n + r).
        self.Z = np.zeros((self.n - 1, 4))

        self.fitted = False
        self.debug = False

        # Timing diagnostics for edge sorting vs. fitting
        self.elapse_sort = 0.0
        self.elapse_fit = 0.0

    # Produces a sorted list of (i, j) pairs representing all edges in the
    # complete graph on n nodes, ordered by their distance d[i,j] ascending.
    # This defines the order in which edges are processed by learn_UM.
    #
    # Parameters:
    #   d    : (n, n) distance matrix
    #   n    : number of nodes
    #   rand : if True, shuffle edges before sorting (breaks ties randomly)
    def get_edge_seq(self, d, n, rand=False):
        entries = []
        edges = []
        # Collect all pairs (i, j) with i > j (lower triangle only) to avoid
        # processing each edge twice
        for i in range(n):
            for j in range(i):
                entries.append((i, j, d[i, j]))
        if rand:
            random.shuffle(entries)
        # Sort by distance so we process nearest pairs first, like Kruskal's algorithm
        entries.sort(key=lambda e: e[2])
        for e in entries:
            edges.append((e[0], e[1]))
        return edges

    # Updates the bookkeeping matrices when edge (i, j) is processed.
    # Increments A, N, H, and M to reflect that leaf i and leaf j have now
    # "seen" each other, potentially triggering majority flags.
    #
    # Parameters:
    #   i, j : leaf node indices whose edge is being processed
    def update_matrices(self, i, j):
        k = self.membership[i]   # current cluster of leaf i
        l = self.membership[j]   # current cluster of leaf j

        # Record that i and j have seen each other
        self.A[i, j] += 1
        self.A[j, i] += 1

        # i has seen one more member of j's cluster, and vice versa
        self.N[i, l] += 1
        self.N[j, k] += 1

        # Only update majority flags if i and j are in different clusters
        # (edges within the same cluster don't contribute to cross-cluster voting)
        if k != l:
            # Has leaf i now seen a majority of cluster l?
            if self.H[i, l] == 0 and 2 * self.N[i, l] >= self.S[l]:
                self.H[i, l] = 1          # mark i as having majority-seen l
                self.M[k, l] += 1         # one more leaf in k has majority-seen l

            # Has leaf j now seen a majority of cluster k?
            if self.H[j, k] == 0 and 2 * self.N[j, k] >= self.S[k]:
                self.H[j, k] = 1          # mark j as having majority-seen k
                self.M[l, k] += 1         # one more leaf in l has majority-seen k

    # Merges clusters k and l into a new internal node r, recording the merge
    # in the linkage matrix Z and the dendrogram graph G.
    #
    # The merge condition has already been verified by learn_UM before this
    # is called: every leaf in k has majority-seen l, and vice versa.
    #
    # Parameters:
    #   k, l     : cluster IDs being merged
    #   distance : the edge distance at which the merge was triggered
    def merge_clusters(self, k, l, distance):
        # Assign the next available internal node ID to the new merged cluster
        r = self.nextroots.pop(-1)
        new_size = self.S[k] + self.S[l]
        self.S[r] = new_size

        # The new cluster has seen everything both child clusters have seen
        self.N[:, r] = self.N[:, k] + self.N[:, l]

        X = []   # leaves currently in cluster k
        Y = []   # leaves currently in cluster l

        for v in range(self.n):
            # Check if any leaf already has majority-seen the new merged cluster r
            if 2 * self.N[v, r] >= new_size:
                self.H[v, r] = 1
                self.M[self.membership[v], r] += 1
            if self.membership[v] == k:
                X.append(v)
            if self.membership[v] == l:
                Y.append(v)

        # The new node inherits the cross-cluster majority counts from both children
        self.M[r, :] = self.M[k, :] + self.M[l, :]

        # Set the ultrametric distance between all cross-cluster leaf pairs to
        # the current merge distance (this is what makes d_U an ultrametric:
        # all pairs that merge at this step get the same distance value)
        for x in X:
            for y in Y:
                self.d_U[x, y] = distance
                self.d_U[y, x] = distance

        # Add the new internal node and its two child edges to the dendrogram graph.
        # Edge lengths are the *difference* in heights (branch length, not total height),
        # so the tree can be plotted as a proper dendrogram.
        self.G.add_node(r)
        self.G.add_edge(r, k, length=distance - self.heights[k])
        self.G.add_edge(r, l, length=distance - self.heights[l])
        self.heights[r] = distance

        # Record the merge in the scipy-format linkage matrix
        self.Z[r - self.n] = np.array([k, l, distance, new_size])

        # Update membership: all leaves in k and l now belong to r
        for x in X:
            self.membership[x] = r
        for y in Y:
            self.membership[y] = r

        if self.debug:
            print(X, Y, distance)

    # Main fitting method. Processes edges in ascending distance order,
    # updating bookkeeping after each edge, and merging clusters whenever
    # the mutual majority condition is satisfied.
    #
    # The mutual majority condition for merging clusters k and l is:
    #   M[k, l] + M[l, k] == S[k] + S[l]
    # which means: every leaf in k has majority-seen l, AND every leaf in l
    # has majority-seen k. This is what enforces the ultrametric property.
    #
    # The algorithm terminates when only one cluster remains (nextroots is
    # exhausted down to 1 entry, meaning n-1 merges have occurred).
    def learn_UM(self):
        start = time.time()
        # Get all edges sorted by distance — O(n² log n)
        E = self.get_edge_seq(self.d, self.n)
        end = time.time()
        self.elapse_sort = end - start

        t = 0   # current edge index
        # Loop until all nodes have been merged into one tree (n-1 merges total)
        while len(self.nextroots) > 1:
            i, j = E[t][0], E[t][1]
            self.update_matrices(i, j)
            k, l = self.membership[i], self.membership[j]
            # Check mutual majority condition: every leaf on both sides has
            # now seen a majority of the other cluster — trigger a merge
            if k != l and self.M[k, l] + self.M[l, k] == self.S[k] + self.S[l]:
                self.merge_clusters(k, l, self.d[i, j])
            t += 1

        self.fitted = True
        end = time.time()
        self.elapse_fit = end - start


def plot_dendrogram(distance_matrix, no_labels=True):
    hcc = HccLinkage(distance_matrix)
    hcc.learn_UM()

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(hcc.Z, ax=ax, no_labels=no_labels)
    ax.set_xlabel("District UID")
    ax.set_ylabel("Merge distance")
    ax.set_title("District clustering dendrogram")
    plt.tight_layout()
    plt.show()