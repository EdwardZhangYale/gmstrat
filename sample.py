from __future__ import annotations

import gzip
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hccfit import HccLinkage
from utils import read_json, str_to_vec, vec_to_str, weighted_l1

# lovely comments by Claude Sonnet 4.6

# A frozen dataclass (immutable after creation) that acts as a namespace for
# all file paths associated with a single processed sample set. Using a
# dataclass here instead of hardcoding paths everywhere makes it easy to
# relocate the storage directory by changing only `root`.
@dataclass(frozen=True)
class SampleStoragePaths:
    root: Path

    # Each property constructs a full path by joining root with a fixed filename.
    # Using .feather files for DataFrames (fast binary columnar format) and
    # .npy files for numpy arrays (fast binary array format).

    @property
    def samples(self) -> Path:
        # Raw sample records: one row per MCMC step (step number, plan vector, tag)
        return self.root / "samples.feather"

    @property
    def plans(self) -> Path:
        # Summarized plan records: one row per unique plan observed
        return self.root / "plans.feather"

    @property
    def districts(self) -> Path:
        # Catalog of all unique districts observed across all samples
        return self.root / "districts.feather"

    @property
    def distributions(self) -> Path:
        # Frequency distribution of plans per sample tag
        return self.root / "distributions.feather"

    @property
    def distance(self) -> Path:
        # Pairwise distance matrix between all unique districts (square, symmetric)
        return self.root / "distance_matrix.npy"

    @property
    def linkage(self) -> Path:
        # Hierarchical clustering linkage matrix (output of HccLinkage)
        return self.root / "linkage.npy"

    @property
    def pdist_edges(self) -> Path:
        # Bin edges for the pairwise distance histogram, used for discretizing distances
        return self.root / "pdist_edges.npy"


# Opens a sample file for reading, transparently handling both plain text
# and gzip-compressed files based on the file extension.
def _open_sample_file(filename, mode="rt"):
    fn = Path(filename)
    if ".gz" in fn.suffixes:
        return gzip.open(fn, mode)
    return open(fn, mode)


class SampleProcessor:
    # Initializes the processor by loading precinct (node) data from a graph
    # JSON file (as produced by generate_grid_graph), and setting up the
    # output directory and internal state.
    def __init__(self, precinct_fn, save_dir):
        precinct_data = read_json(precinct_fn)
        self.num_districts = int(precinct_data["num_districts"])

        # Build a DataFrame of precinct attributes from the graph's node list
        df_precincts = pd.DataFrame(precinct_data["nodes"]).reset_index(drop=True)
        self.df_precincts = df_precincts
        self.num_precincts = len(df_precincts)

        # Store population as a numpy int32 array for efficient numeric operations
        self.population = np.asarray(
            df_precincts["population"].to_numpy(), dtype=np.int32
        )
        self.total_pop = int(self.population.sum(dtype=np.int32))

        # Build a lookup dictionary mapping precinct string labels (e.g. "(0,1)")
        # to their integer row indices in df_precincts, for fast plan vector parsing
        precinct_keys = df_precincts["precinct_id_str"].astype(str).to_list()
        self.prec_to_idx = {key: idx for idx, key in enumerate(precinct_keys)}

        # The maximum meaningful distance between two districts: set to 2 * total_pop / num_districts.
        # This is the L1 population distance between two districts that are
        # exact complements of each other (one contains all of one half of the
        # population, the other contains all of the other half).
        self.maximum_distance = int((self.total_pop * 2) // self.num_districts)

        # Set up output directory structure and create it if it doesn't exist
        self.paths = SampleStoragePaths(root=Path(save_dir))
        self.paths.root.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded cache of all district vectors (populated on first access)
        self.all_districts = None

    # Wipes and recreates the output directory, discarding all previously
    # processed data. Useful for reprocessing from scratch.
    def clean(self):
        shutil.rmtree(self.paths.root, ignore_errors=True)
        self.paths.root.mkdir(parents=True, exist_ok=True)

    # Loads previously processed results from disk back into memory, avoiding
    # the need to re-parse and re-process raw sample files. Also recomputes
    # or loads the pairwise distance bin edges.
    def load_processed(self):
        self.df_plans = pd.read_feather(self.paths.plans)
        self.df_districts = pd.read_feather(self.paths.districts)
        self.df_samples = pd.read_feather(self.paths.samples)
        self.df_distributions = pd.read_feather(self.paths.distributions)
        self.num_districts = len(self.df_plans.plan.iloc[0])
        self.maximum_distance = int((self.total_pop * 2) // self.num_districts)
        try:
            # Try to load precomputed bin edges from disk
            self.pdist_edges = np.load(self.paths.pdist_edges, allow_pickle=False)
        except FileNotFoundError:
            # If not found, compute and save them
            self.pdist_edges = self._compute_pdist_edges()
            np.save(self.paths.pdist_edges, self.pdist_edges, allow_pickle=False)
        self.all_districts = None

    # Main entry point for processing raw sample files end-to-end.
    # Reads samples, expands them into per-district rows, builds a district
    # catalog, summarizes plans, calculates frequency distributions, and
    # writes everything to disk.
    #
    # Parameters:
    #   samples_fns        : iterable of file paths to raw JSONL sample files
    #   max_length         : maximum number of samples to read per file
    #   min_step           : skip samples at or before this MCMC step number
    #                        (useful for discarding burn-in)
    #   pdist_edge_samples : number of districts to sample when computing bin edges
    #   pdist_edge_bins    : number of quantile bins for the distance histogram
    def process_samples(
        self,
        samples_fns,
        max_length,
        min_step: int = 0,
        *,
        pdist_edge_samples: int = 100,
        pdist_edge_bins: int = 50,
    ):
        df_samples = self._read_samples(samples_fns, max_length, min_step)
        df_samples_expanded = self._expand_samples(df_samples)
        df_districts = self._build_district_catalog(df_samples_expanded)
        df_samples_expanded = self._attach_district_ids(
            df_samples_expanded, df_districts
        )
        df_plans = self._summarize_plans(df_samples_expanded)
        df_distributions = self._calculate_distributions(df_plans)

        # Store all results on self for downstream access
        self.df_samples = df_samples
        self.df_districts = df_districts
        self.df_plans = df_plans
        self.df_distributions = df_distributions

        # Persist all DataFrames and arrays to disk
        self._write_feather(df_samples, self.paths.samples)
        self._write_feather(df_districts, self.paths.districts)
        self._write_feather(df_plans, self.paths.plans)
        self._write_feather(df_distributions, self.paths.distributions)
        self.pdist_edges = self._compute_pdist_edges(
            n_samples=pdist_edge_samples,
            n_bins=pdist_edge_bins,
        )
        np.save(self.paths.pdist_edges, self.pdist_edges, allow_pickle=False)

        # Reset the district cache since districts may have changed
        self.all_districts = None

    # Reads raw JSONL sample files and returns a DataFrame with one row per
    # valid MCMC sample. Handles the file format described earlier:
    #   line 0: plain string header (skipped via line_idx < 3)
    #   line 1: metadata (skipped)
    #   line 2: parameters dict — parsed to extract num_districts if present
    #   line 3+: sample records with "name" and "districting" keys
    def _read_samples(
        self, samples_fns: Iterable[Path | str], max_length: int, min_step: int
    ):
        records = []
        for fn in samples_fns:
            path = Path(fn)
            print(f"Reading {path}")
            # Use the file stem (before first ".") as a human-readable tag
            # to distinguish samples from different runs
            sample_tag = path.stem.split(".")[0]
            file_count = 0
            with _open_sample_file(path) as handle:
                with tqdm(
                    total=max_length, desc=f"{sample_tag}", unit="samples"
                ) as pbar:
                    for line_idx, line in enumerate(handle):
                        if file_count >= max_length:
                            break

                        # Line 2 is the parameters dict — extract num_districts
                        # and recompute maximum_distance if available
                        if line_idx == 2:
                            header = json.loads(line)
                            if isinstance(header, dict) and "districts" in header:
                                self.num_districts = int(header["districts"])
                                self.maximum_distance = int(
                                    (self.total_pop * 2) // self.num_districts
                                )

                        # Skip the first 3 header/metadata lines
                        if line_idx < 3:
                            continue

                        data = json.loads(line)

                        # Parse the step number from the "name" field (e.g. "step1000" → 1000)
                        step = int(str(data["name"]).removeprefix("step"))

                        # Skip burn-in samples at or before min_step
                        if step <= min_step:
                            continue

                        # Parse the districting into a 0-indexed integer array
                        # (subtract 1 because districts in the file are 1-indexed)
                        plan_vector = self._parse_plan_vector(data["districting"]) - 1
                        records.append(
                            {
                                "step": step,
                                "plan_vector": plan_vector,
                                "sample_tag": sample_tag,
                            }
                        )
                        file_count += 1
                        pbar.update(1)
        return pd.DataFrame.from_records(records)

    # Converts a raw "districting" list of dicts (as stored in the JSONL file)
    # into a numpy array of length num_precincts, where entry i is the district
    # number assigned to precinct i.
    #
    # The key format in the file can be either a plain string like "(0,1)" or
    # a JSON-encoded list like '["(0,1)"]' — both are handled here.
    def _parse_plan_vector(self, districting: Sequence[dict]) -> np.ndarray:
        plan_vector = np.zeros(self.num_precincts, dtype=np.int32)
        for assignment in districting:
            key = next(iter(assignment.keys()))
            precinct = key
            # Unwrap JSON-encoded list keys like '["(0,1)"]' → "(0,1)"
            if isinstance(key, str) and key.startswith("[") and key.endswith("]"):
                try:
                    precinct = json.loads(key)[0]
                except Exception:
                    precinct = key
            # Look up the precinct's integer index and record its district assignment
            plan_vector[self.prec_to_idx[str(precinct)]] = int(assignment[key])
        return plan_vector

    # Expands the samples DataFrame from one row per plan to one row per
    # (plan, district) pair. For each plan, iterates over all districts and
    # records which precincts belong to that district as an array of indices.
    #
    # Input:  df_samples      — one row per MCMC sample, with a plan_vector column
    # Output: df_expanded     — one row per (sample, district), with district_vector
    #                           as a sorted array of precinct indices in that district
    def _expand_samples(self, df_samples: pd.DataFrame) -> pd.DataFrame:
        expanded_rows = []
        for plan_id, row in tqdm(
            df_samples.iterrows(), total=len(df_samples), desc="plans"
        ):
            pvec = row.plan_vector
            for district_id in range(self.num_districts):
                # np.where returns indices where the condition is True —
                # i.e. the set of precincts assigned to this district
                dvec = np.where(pvec == district_id)[0]
                expanded_rows.append(
                    {
                        "step": row.step,
                        "sample_tag": row.sample_tag,
                        "plan_id": plan_id,
                        "district_id": district_id,
                        "district_vector": dvec,
                    }
                )
        return pd.DataFrame.from_records(expanded_rows)

    # Builds a deduplicated catalog of all unique districts observed across
    # all samples. Each unique district is assigned a stable integer ID
    # (district_uid) based on its position in the deduplicated list.
    #
    # Districts are compared by their string representation (via vec_to_str)
    # to allow deduplication with pandas drop_duplicates.
    def _build_district_catalog(
        self, df_samples_expanded: pd.DataFrame
    ) -> pd.DataFrame:
        # Serialize each district_vector to a canonical string for deduplication
        df_samples_expanded["district_str"] = df_samples_expanded.district_vector.apply(
            vec_to_str
        )
        # Keep only the first occurrence of each unique district string
        df_districts = df_samples_expanded.drop_duplicates(subset=["district_str"])[
            ["district_str"]
        ].reset_index(drop=True)
        # Assign a stable integer UID equal to the row index
        df_districts["district_uid"] = df_districts.index
        return df_districts

    # Joins the district catalog back onto the expanded samples DataFrame,
    # replacing each district_vector with its integer district_uid.
    # This is a standard encode step: replace large objects with compact IDs.
    def _attach_district_ids(
        self, df_samples_expanded: pd.DataFrame, df_districts: pd.DataFrame
    ):
        # Build a dict mapping district_str → district_uid for fast lookup
        district_map = dict(zip(df_districts.district_str, df_districts.district_uid))
        df_samples_expanded = df_samples_expanded.copy()
        df_samples_expanded["district_uid"] = df_samples_expanded.district_str.map(
            district_map
        )
        return df_samples_expanded

    # Collapses the expanded (plan, district) rows back to one row per plan.
    # Each plan is represented as a sorted list of district_uids, and also
    # as a canonical dot-separated string (e.g. "0.3.7") for deduplication.
    def _summarize_plans(self, df_samples_expanded: pd.DataFrame) -> pd.DataFrame:
        # Group by plan identity and collect the list of district UIDs per plan
        df_plans = (
            df_samples_expanded.groupby(["step", "plan_id", "sample_tag"])[
                "district_uid"
            ]
            .apply(list)
            .reset_index(name="plan")
        )
        # Create a canonical string representation for each plan by sorting UIDs
        # (so that plans with the same districts in different orders are treated as equal)
        df_plans["plan_str"] = df_plans.plan.apply(
            lambda x: ".".join([str(i) for i in sorted(x)])
        )
        return df_plans

    # Computes how often each distinct plan appears in each sample run,
    # both as a raw count and as a relative frequency. This is the empirical
    # distribution over plans produced by the MCMC chain.
    def _calculate_distributions(self, df_plans: pd.DataFrame) -> pd.DataFrame:
        # Count occurrences of each (sample_tag, plan_str) pair
        df_distributions = (
            df_plans.groupby(["sample_tag", "plan_str"])
            .size()
            .reset_index(name="count")
        )
        # Normalize counts within each sample_tag to get relative frequencies
        df_distributions["freq"] = df_distributions.groupby("sample_tag")[
            "count"
        ].transform(lambda x: x / x.sum())
        # Decode plan_str back to a vector for downstream use
        df_distributions["plan_vector"] = df_distributions.plan_str.apply(str_to_vec)
        return df_distributions

    # Writes a DataFrame to disk in the Feather binary format.
    # Feather is a fast, language-agnostic columnar format optimized for
    # read/write speed — much faster than CSV for large DataFrames.
    def _write_feather(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(path)

    # Computes bin edges for a histogram of pairwise district distances.
    # Rather than computing all O(n²) pairwise distances (expensive for large
    # district catalogs), this samples a subset of districts and computes
    # distances only within that subset.
    #
    # Uses quantile-based binning (pd.qcut) so that each bin contains roughly
    # equal numbers of observations, giving better resolution where distances
    # are dense.
    #
    # Parameters:
    #   n_samples : number of districts to randomly sample
    #   n_bins    : number of quantile bins to create
    def _compute_pdist_edges(
        self, n_samples: int = 500, n_bins: int = 400
    ) -> np.ndarray:
        df = self.df_districts
        # Cap n_samples at the actual number of districts available
        n = min(int(n_samples), int(len(df)))
        sampled = df.sample(n=n, random_state=0)  # fixed seed for reproducibility
        districts = sampled["district_str"].apply(str_to_vec).to_list()

        # Compute all pairwise distances in the sampled subset (lower triangle only)
        dist_list = []
        for i in range(n):
            for j in range(i):
                d = int(self.compute_distance(districts[i], districts[j]))
                # Exclude pairs beyond maximum_distance (they'll be in their own bin)
                if d < self.maximum_distance:
                    dist_list.append(d)

        # Add sentinel values to ensure the full range [-1, maximum_distance+1]
        # is covered, preventing qcut from failing on edge cases
        right_edge = self.maximum_distance + 1
        _, edges = pd.qcut(
            dist_list + [-1, right_edge],
            q=int(n_bins),
            retbins=True,       # return the bin edges, not the binned values
            duplicates="drop",  # drop duplicate edges (can occur when data is sparse)
        )
        # Return deduplicated edges as int32 array
        return np.unique(np.asarray(edges, dtype=np.int32))

    # Computes the weighted L1 distance between two districts x and y.
    # Each district is represented as a sorted array of precinct indices.
    # The distance is weighted by precinct population, so swapping a high-
    # population precinct contributes more to the distance than a low-population one.
    def compute_distance(self, x, y):
        return weighted_l1(x, y, self.population, self.maximum_distance, sparse=True)

    # Computes the full pairwise distance matrix for a list of districts.
    # Returns a symmetric N×N matrix where entry [i,j] is the distance
    # between district i and district j. Only the lower triangle is computed;
    # the upper triangle is filled by symmetry.
    def compute_distance_matrix(self, districts):
        num_districts = len(districts)
        distance_matrix = np.zeros((num_districts, num_districts), dtype=np.int32)
        total = num_districts * (num_districts - 1) // 2  # number of unique pairs
        pbar = tqdm(total=total)
        for i in range(num_districts):
            for j in range(i):
                d = int(self.compute_distance(districts[i], districts[j]))
                # Fill both [i,j] and [j,i] to maintain symmetry
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d
                pbar.update(1)
        return distance_matrix

    # Loads the distance matrix from disk if it exists, otherwise computes
    # it from scratch and saves it. This is a standard compute-once-cache pattern.
    def load_distance_matrix(self):
        if not self.paths.distance.is_file():
            distance_matrix = self.compute_distance_matrix(self.get_all_districts())
            np.save(self.paths.distance, distance_matrix)
        return np.load(self.paths.distance, allow_pickle=False)

    # Ensures the hierarchical clustering linkage matrix exists on disk.
    # If it doesn't, computes the distance matrix, runs HccLinkage to learn
    # the ultrametric linkage (hcc.learn_UM()), and saves the result.
    # This is used downstream for clustering districts into groups.
    def ensure_linkage(self) -> None:
        if self.paths.linkage.is_file():
            return  # already computed, nothing to do
        distance_matrix = self.load_distance_matrix()
        hcc = HccLinkage(distance_matrix)
        hcc.learn_UM()  # fit the ultrametric hierarchical clustering
        np.save(self.paths.linkage, hcc.Z.astype(np.int32))

    # Returns the full list of district vectors (as numpy arrays of precinct indices)
    # for all unique districts in the catalog. Uses lazy initialization: computes
    # and caches on first call, returns cached value on subsequent calls.
    def get_all_districts(self):
        if self.all_districts is None:
            self.all_districts = self.df_districts.district_str.apply(
                str_to_vec
            ).to_list()
        return self.all_districts

    # Looks up a district vector in the catalog and returns its integer UID.
    # This is the inverse of decode_district.
    def encode_district(self, district):
        return (
            self.df_districts[self.df_districts.district_str == vec_to_str(district)]
            .iloc[0]
            .district_uid
        )

    # Returns the district vector (array of precinct indices) for a given
    # district UID. This is the inverse of encode_district.
    def decode_district(self, district_uid):
        return self.get_all_districts()[district_uid]

    # Converts a district (array of precinct indices) into a dense binary
    # vector of length num_precincts, where entry i is 1 if precinct i is
    # in the district and 0 otherwise.
    def decode_district_vector(self, district):
        district_vector = np.zeros(self.num_precincts, dtype=np.int32)
        district_vector[district] += 1
        return district_vector

    # Converts a plan (list of district UIDs) back into a full plan vector —
    # an array of length num_precincts where entry i is the district ID
    # (0-indexed) assigned to precinct i.
    def decode_plan_vector(self, plan):
        all_districts = self.get_all_districts()
        plan_vector = np.zeros(self.num_precincts, dtype=np.int32)
        for district_id, district_uid in enumerate(plan):
            for precinct in all_districts[district_uid]:
                plan_vector[precinct] = district_id
        return plan_vector