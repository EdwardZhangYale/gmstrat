from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from utils import str_to_vec

# Annotations by Claude Sonnet 4.6

# Extracts a flat clustering of `num_vertices` items from a hierarchical
# linkage matrix, stopping early once the desired number of clusters k is
# reached.
#
# A linkage matrix (as produced by HccLinkage) encodes a dendrogram: each row
# says "merge node x and node y into a new node". Leaf nodes are 0..N-1;
# internal (merged) nodes are N, N+1, N+2, ... The algorithm replays these
# merges until only k groups remain.
#
# Returns:
#   c2i : dict mapping cluster_id (int) → list of member district UIDs
#   i2c : array of length num_vertices where i2c[district_uid] = cluster_id
def get_cluster_from_linkage(
    k: int, linkage: np.ndarray
) -> Tuple[Dict[int, List[int]], np.ndarray]:
    num_vertices = len(linkage) + 1  # N merges → N+1 leaves

    # Start with each node in its own singleton cluster, keyed by node ID
    membership = {i: [i] for i in range(num_vertices)}

    for merge_idx, merge in enumerate(linkage[:, :2], start=0):
        x, y = int(merge[0]), int(merge[1])
        # Create a new internal node that absorbs the two merged nodes,
        # removing them from the active membership dict
        membership[merge_idx + num_vertices] = membership.pop(x) + membership.pop(y)
        # Stop as soon as we've reached the desired number of clusters
        if len(membership) == k:
            break

    # Relabel remaining membership groups as contiguous cluster IDs 0..k-1
    c2i: Dict[int, List[int]] = defaultdict(list)
    i2c = np.zeros(num_vertices, dtype=np.int32)
    for cluster_id, cluster_name in enumerate(list(membership.keys())):
        for member in membership[cluster_name]:
            c2i[cluster_id].append(member)
            i2c[member] = cluster_id
    return c2i, i2c


class HClusters:
    # Wraps a SampleProcessor (sp) and provides hierarchical + k-centroid
    # clustering over the district catalog it contains.
    #
    # Parameters:
    #   sp : a SampleProcessor instance with a fitted linkage matrix on disk
    def __init__(self, sp):
        self.sp = sp
        self.linkage = np.load(sp.paths.linkage)   # the HCC linkage matrix
        self.num_vertices = len(self.linkage) + 1  # total number of unique districts
        self.centroids: List[np.ndarray] = []      # one centroid per cluster (list of precinct indices)
        self.cluster_densities: np.ndarray | None = None  # shape (num_clusters, num_precincts)

    # Cuts the dendrogram at a level that produces `num_clusters` clusters,
    # then optionally computes centroids for each cluster.
    def update_clusters(self, num_clusters: int, update_centroids: bool = True) -> None:
        self.num_clusters = num_clusters
        self.c2i, self.i2c = get_cluster_from_linkage(self.num_clusters, self.linkage)
        if update_centroids:
            self.update_centroids()

    # Refines the initial hierarchical clustering using a k-centroid iteration
    # (analogous to k-means, but using weighted L1 distance instead of
    # Euclidean). Runs until assignments stop changing, WCSS stops improving,
    # or max_iter is reached.
    #
    # Parameters:
    #   convergence_thres : stop if relative WCSS improvement drops below this
    #   max_iter          : hard cap on number of refinement epochs
    #   verbose           : print per-epoch improvement if True
    def kcentroids(
        self, convergence_thres: float = 0.01, max_iter: int = 50, verbose=True
    ) -> None:
        num_clusters = int(getattr(self, "num_clusters", 0))
        all_districts = self.sp.get_all_districts()

        # Snapshot current assignments and WCSS as the baseline for comparison
        prev_assignments = np.asarray(self.i2c, dtype=np.int32).copy()
        prev_wcss = self._compute_wcss_centroid(all_districts, prev_assignments)

        for epoch in tqdm(range(max_iter)):
            # E-step: reassign each district to its nearest centroid
            new_assignments = self._assign_to_nearest_centroids(
                all_districts,
                centroids=self.centroids,
                previous_assignments=prev_assignments,
                num_clusters=num_clusters,
            )
            assignments_unchanged = np.array_equal(new_assignments, prev_assignments)

            # M-step: update centroids based on new assignments
            self.i2c = new_assignments
            self.c2i = self._build_c2i_from_assignments(new_assignments, num_clusters)
            self.update_centroids()

            new_wcss = self._compute_wcss_centroid(all_districts, new_assignments)
            improvement = prev_wcss - new_wcss

            # If WCSS got worse (can happen due to centroid discretization),
            # stop immediately
            if improvement <= 0.0:
                break

            rel_improvement = improvement / prev_wcss if prev_wcss > 0.0 else 0.0
            prev_assignments = new_assignments
            prev_wcss = new_wcss

            # Converged: assignments are stable or improvement is negligible
            if assignments_unchanged or (rel_improvement < convergence_thres):
                break
            if verbose:
                print(f"epoch {epoch}: rel_improvement = {rel_improvement}")

    # Builds the c2i (cluster → list of district UIDs) dict from a flat
    # assignments array. Inverse of i2c.
    def _build_c2i_from_assignments(
        self, assignments: np.ndarray, num_clusters: int
    ) -> Dict[int, List[int]]:
        c2i: Dict[int, List[int]] = defaultdict(list)
        for district_uid, cluster_id in enumerate(assignments):
            cid = int(cluster_id)
            c2i[cid].append(int(district_uid))
        return c2i

    # Assigns each district to its nearest centroid by weighted L1 distance.
    # Uses a vectorized intersection trick to avoid looping over centroids:
    # for sparse sets A and B with population weights w:
    #   weighted_L1(A, B) = sum_w(A) + sum_w(B) - 2 * sum_w(A ∩ B)
    # where sum_w(A ∩ B) = centroid_members[c, :] @ weights (dot product over
    # the district's precinct indices).
    #
    # Tie-breaking: if the current assignment is among the nearest centroids,
    # keep it (stability preference).
    def _assign_to_nearest_centroids(
        self,
        districts: Sequence[np.ndarray],
        *,
        centroids: Sequence[np.ndarray],
        previous_assignments: np.ndarray,
        num_clusters: int,
    ) -> np.ndarray:
        weights = np.asarray(self.sp.population, dtype=np.int64)
        max_dist = (
            None
            if self.sp.maximum_distance is None
            else int(np.int64(self.sp.maximum_distance))
        )

        # Precompute a boolean membership matrix for all centroids:
        # centroid_members[c, p] = True if precinct p is in centroid c
        centroid_members = np.zeros(
            (num_clusters, self.sp.num_precincts), dtype=np.bool_
        )
        # Also precompute the total population weight of each centroid
        centroid_weight_sum = np.zeros(num_clusters, dtype=np.int64)
        for cluster_id, centroid in enumerate(centroids):
            idx = np.asarray(centroid, dtype=np.intp)
            if idx.size:
                centroid_members[cluster_id, idx] = True
                centroid_weight_sum[cluster_id] = weights[idx].sum(dtype=np.int64)

        num_points = len(districts)
        assignments = np.empty(num_points, dtype=np.int32)

        for district_uid, district in enumerate(districts):
            idx = np.asarray(district, dtype=np.intp)
            if idx.size:
                w_idx = weights[idx]
                sum_w = int(w_idx.sum(dtype=np.int64))
                # Compute intersection weight with every centroid simultaneously:
                # centroid_members[:, idx] is shape (num_clusters, len(idx)),
                # @ w_idx gives a (num_clusters,) vector of intersection weights
                inter = centroid_members[:, idx] @ w_idx
            else:
                sum_w = 0
                inter = np.zeros(num_clusters, dtype=np.int64)

            # Weighted L1 distance to every centroid in one vectorized expression
            dist_vec = sum_w + centroid_weight_sum - 2 * inter
            if max_dist is not None:
                dist_vec = np.minimum(dist_vec, max_dist)  # cap at maximum_distance

            min_dist = int(dist_vec.min())
            candidates = np.flatnonzero(dist_vec == min_dist)

            # Prefer the current assignment if it's tied for nearest (stability)
            current = int(previous_assignments[district_uid])
            if np.any(candidates == current):
                chosen = current
            else:
                chosen = int(candidates[0])

            assignments[district_uid] = chosen

        return assignments

    # Computes the Within-Cluster Sum of Squares (WCSS) using the same
    # vectorized weighted L1 formula as _assign_to_nearest_centroids, but
    # only against each district's *assigned* centroid rather than all of them.
    # Lower WCSS = tighter, more compact clusters.
    def _compute_wcss_centroid(
        self, districts: Sequence[np.ndarray], assignments: np.ndarray
    ) -> float:
        num_clusters = int(getattr(self, "num_clusters", 0))
        weights = np.asarray(self.sp.population, dtype=np.int64)
        max_dist = (
            None
            if self.sp.maximum_distance is None
            else int(np.int64(self.sp.maximum_distance))
        )

        # Same centroid precomputation as in _assign_to_nearest_centroids
        centroid_members = np.zeros(
            (num_clusters, self.sp.num_precincts), dtype=np.bool_
        )
        centroid_weight_sum = np.zeros(num_clusters, dtype=np.int64)
        for cluster_id, centroid in enumerate(self.centroids):
            idx = np.asarray(centroid, dtype=np.intp)
            if idx.size:
                centroid_members[cluster_id, idx] = True
                centroid_weight_sum[cluster_id] = weights[idx].sum(dtype=np.int64)

        total = np.int64(0)
        for district_uid, district in enumerate(districts):
            cluster_id = int(assignments[district_uid])
            idx = np.asarray(district, dtype=np.intp)
            if idx.size:
                w_idx = weights[idx]
                sum_w = int(w_idx.sum(dtype=np.int64))
                inter = int(centroid_members[cluster_id, idx] @ w_idx)
            else:
                sum_w = 0
                inter = 0
            # Weighted L1 distance between this district and its cluster centroid
            dist = sum_w + int(centroid_weight_sum[cluster_id]) - 2 * inter
            if max_dist is not None and dist > max_dist:
                dist = max_dist
            total += dist
        return float(total)

    # Recomputes centroids and cluster density maps from current assignments.
    #
    # The centroid of a cluster is defined as the set of precincts that appear
    # in at least 50% of the cluster's member districts. This is a majority-vote
    # aggregation — a natural analog of the k-means mean centroid, but for
    # sets rather than vectors.
    #
    # Also stores cluster_densities: a (num_clusters, num_precincts) array where
    # entry [c, p] is the fraction of cluster c's districts that contain precinct p.
    def update_centroids(self) -> None:
        all_districts = self.sp.get_all_districts()
        centroids: List[np.ndarray] = []
        densities: List[np.ndarray] = []

        num_clusters = int(getattr(self, "num_clusters", max(self.i2c) + 1))
        for cluster_id in range(num_clusters):
            members = self.c2i[cluster_id]
            cls_size = len(members)
            if cls_size == 0:
                # Empty cluster: zero density, empty centroid
                densities.append(np.zeros(self.sp.num_precincts, dtype=float))
                centroids.append(np.array([], dtype=int))
                continue

            # Count how many districts in this cluster contain each precinct
            counts = np.zeros(self.sp.num_precincts, dtype=np.int32)
            for district_uid in members:
                counts[all_districts[district_uid]] += 1

            # Density = fraction of cluster members containing each precinct
            dens = counts / cls_size
            # Centroid = precincts present in >= 50% of cluster members
            centroid = np.flatnonzero(dens >= 0.5)
            densities.append(dens)
            centroids.append(centroid)

        self.cluster_densities = np.vstack(densities)
        self.centroids = centroids

    # Computes a per-district WCSS breakdown as a DataFrame, showing each
    # district's assigned cluster, its centroid, and its distance to that
    # centroid. Useful for inspecting cluster quality at the individual
    # district level rather than as a single aggregate score.
    def compute_wcss(self):
        df = self.sp.df_districts.copy()
        df["dvec"] = df.district_str.apply(str_to_vec)          # decode district vector
        df["centroid"] = df.district_uid.apply(lambda x: self.centroids[self.i2c[x]])  # look up centroid
        df["letter"] = df.district_uid.apply(lambda x: self.i2c[x])                   # cluster label
        df["dcentroid"] = df.apply(
            lambda row: self.sp.compute_distance(row.dvec, row.centroid), axis=1       # distance to centroid
        )
        return df