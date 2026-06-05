from __future__ import annotations

import json
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors


def read_json(fn):
    with open(fn, "r") as f:
        return json.load(f)


# Encodes a district (a collection of precinct indices) as a canonical
# dot-separated string, e.g. [3, 1, 2] → "1.2.3".
# Sorting ensures the representation is order-independent, so two districts
# containing the same precincts always produce the same string key.
# Used to deduplicate districts across sampled plans.
def vec_to_str(vec):
    return ".".join([str(int(i)) for i in sorted(vec)])


# Decodes a dot-separated string back into a numpy array of precinct indices.
# The inverse of vec_to_str. An empty string maps to an empty array,
# representing a district with no precincts (e.g. a sentinel/null district).
def str_to_vec(s):
    if s == "":
        return np.array([], dtype=np.int32)
    return np.array(s.split("."), dtype=np.int32)


# Computes the population-weighted L1 distance between two districts.
#
# Conceptually: two districts are "close" if the population living in
# precincts that one contains but the other doesn't is small.
#
# Two calling modes:
#
#   sparse=True  : v1 and v2 are lists of precinct *indices* (the sparse
#                  representation used throughout the pipeline). The
#                  symmetric difference of the two index sets identifies
#                  precincts in exactly one district; their weights are summed.
#
#   sparse=False : v1 and v2 are dense binary vectors of length P (one entry
#                  per precinct). The weighted L1 norm of their difference
#                  gives the same quantity but via a dot product.
#
# In both modes the result is capped at `maximum_distance` to prevent two
# completely disjoint districts from dominating the geometry of the space
# (as discussed in the paper's distance definition).
#
# Returns np.int32 in sparse mode, float in dense mode.
def weighted_l1(
    v1, v2, weight, maximum_distance=np.iinfo(np.int32).max, *, sparse: bool
):
    if sparse:
        w = np.asarray(weight)
        # xor1d gives the symmetric difference: precincts in v1 but not v2,
        # plus precincts in v2 but not v1.
        diff = np.setxor1d(np.asarray(v1, dtype=np.intp), np.asarray(v2, dtype=np.intp))
        dist = w[diff].sum(dtype=np.int64)
        if maximum_distance is not None:
            dist = min(dist, np.int64(maximum_distance))
        return np.int32(dist)

    # Dense mode: straightforward weighted absolute difference.
    v1_f = np.asarray(v1, dtype=float)
    v2_f = np.asarray(v2, dtype=float)
    w_f = np.asarray(weight, dtype=float)
    dist = float(np.dot(np.abs(v1_f - v2_f), w_f))
    if maximum_distance is not None:
        dist = min(dist, float(maximum_distance))
    return dist


# Renders a full redistricting plan as a choropleth map.
#
# Each precinct polygon in `gdf` is colored by its district label in `labels`
# (a length-P array of integers 0..I-1). The `tab20` colormap gives up to 20
# visually distinct district colors. Black edges at linewidth 0.2 delineate
# precinct boundaries within each district.
#
# Useful for quick visual inspection of a single sampled plan.
def plot_plan(gdf, labels, ax=None, cmap="tab20"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    gdf.assign(cluster=labels).plot(
        column="cluster", cmap=cmap, ax=ax, edgecolor="black", linewidth=0.2
    )
    ax.axis("off")
    return ax


# Renders a single district within the state geography.
#
# `mask` can be either:
#   - A boolean array of length P (True = precinct is in this district), or
#   - An array of precinct indices (converted to boolean internally)
#
# Precincts NOT in the district are shown in light grey; precincts IN the
# district are shown in `color` (default red). This is the visualization
# referenced in Figure 2 of the paper, where individual districts from a
# cluster are highlighted in red against a grey state background.
def plot_district(gdf, mask, ax=None, color="red"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    mask_arr = np.asarray(mask)
    if mask_arr.dtype.kind in "iu" and mask_arr.ndim == 1 and mask_arr.size != len(gdf):
        # Input is a list of precinct indices rather than a boolean mask —
        # convert to boolean.
        sel = np.zeros(len(gdf), dtype=bool)
        sel[mask_arr] = True
    else:
        sel = mask_arr.astype(bool)

    gdf.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.2)
    gdf[sel].plot(ax=ax, color=color, edgecolor="black", linewidth=0.5)
    ax.axis("off")
    return ax


# Renders a per-precinct scalar field as a heatmap over the state geography.
#
# `values` is a length-P array of floats — typically cluster_densities[k],
# the fraction of districts in cluster k that contain each precinct. Precincts
# are colored on a continuous scale from vmin to vmax using `cmap` (default
# "Blues"). Darker = higher density = that precinct is more consistently
# present in the cluster's districts.
#
# This is the primary visualization for a single letter: the density map shows
# the "core" of the cluster (dark blue) and its fuzzy boundary (light blue),
# while the letter itself (majority-vote centroid) corresponds to the set of
# precincts with density >= 0.5.
#
# Parameters:
#   colorbar : if True, adds a colorbar legend to the axis
#   vmin/vmax: shared across subplots in plot_words_list so that color scales
#              are comparable between letters in the same word
def plot_distribution(
    gdf,
    values,
    *,
    ax=None,
    cmap="Blues",
    axis=False,
    colorbar=False,
    vmin=None,
    vmax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    gdf.assign(_value=np.asarray(values, dtype=float)).plot(
        column="_value",
        cmap=cmap,
        ax=ax,
        edgecolor="black",
        linewidth=0.2,
        legend=bool(colorbar),
        vmin=vmin,
        vmax=vmax,
    )
    if not axis:
        ax.axis("off")
    return ax


# Visualizes a word as a grid of individual letter density maps.
#
# A word is an I-tuple of letter (cluster) IDs. This function produces one
# subplot per letter, each showing the density map of that cluster —
# i.e. how consistently each precinct appears across the cluster's member
# districts (darker = more consistent inclusion).
#
# Layout:
#   - Subplots arranged in a grid with `cols` columns (default 3)
#   - All subplots share the same color scale (vmin/vmax computed over the
#     entire word), making relative density directly comparable across letters
#   - Each subplot is titled "letter <position>: <cluster_id>", where
#     <position> is the 0-indexed slot in the word and <cluster_id> is the
#     global cluster index in the alphabet
#   - A single shared colorbar is added on the right if show_colorbar=True
#   - Any unused subplot slots (when n_letters % cols != 0) are hidden
#
# Interpretation: this plot answers "what does each district-type in this
# prototypical plan look like?" For a 5-district word in Connecticut, you
# would see 5 subplots each showing the geographic footprint of one district
# type, with intensity indicating how consistently precincts are included.
def plot_words_list(
    gdf,
    cluster_densities,
    word,
    *,
    cluster_sizes=None,
    cmap="Blues",
    cols=3,
    figsize=(10, 4),
    show_colorbar=True,
):
    word = np.asarray(word, dtype=np.int32)
    cluster_densities = np.asarray(cluster_densities, dtype=float)
    n_letters = int(len(word))

    cols = int(min(int(cols), max(n_letters, 1)))
    rows = int(math.ceil(n_letters / cols)) if n_letters else 1
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    # Compute shared color scale across all letters in this word so that
    # density magnitudes are directly comparable between subplots.
    selected = cluster_densities[word] if n_letters else np.asarray([0.0])
    vmin = float(np.min(selected))
    vmax = float(np.max(selected))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9  # guard against degenerate single-value scale

    for idx, centroid_id in enumerate(word.tolist()):
        # cluster_densities[centroid_id] is a length-P vector: entry p is the
        # fraction of districts in cluster centroid_id that contain precinct p.
        values = cluster_densities[int(centroid_id)]
        plot_distribution(
            gdf,
            values,
            ax=axes_flat[idx],
            cmap=cmap,
            axis=True,
            colorbar=False,
            vmin=vmin,
            vmax=vmax,
        )

        if cluster_sizes is not None:
            n = cluster_sizes[int(centroid_id)]
            axes_flat[idx].set_title(f"letter {idx}: {int(centroid_id)}\n(n={n})")
        else:
            axes_flat[idx].set_title(f"letter {idx}: {int(centroid_id)}")

    # Hide any empty axes in the last row of the grid
    for ax in axes_flat[n_letters:]:
        ax.axis("off")

    # Add a single shared colorbar for the whole figure, attached to the
    # non-empty axes only. Colorbar range runs from the minimum to maximum
    # density value across all letters in this word.
    if show_colorbar and n_letters:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(
            sm,
            ax=axes_flat[:n_letters].tolist(),
            fraction=0.035,
            pad=0.04,
            location="right",
        )

    return fig, axes


# Visualizes a word as a single combined map, overlaying all letters
# simultaneously using semi-transparent color blending.
#
# Each letter is assigned a distinct color from `cmap` (default "tab10").
# For each precinct, each letter's color is painted at an alpha proportional
# to the letter's density at that precinct: alpha = density / max_density.
# Letters are drawn in order, so later letters visually overwrite earlier ones
# in high-density regions; precincts with low density in all letters remain
# close to the grey base layer.
#
# The `ltr` (left-to-right) flag, if True, reorders letters by the
# density-weighted mean x-coordinate of their precincts before plotting.
# This makes the color assignment geographically intuitive — leftmost
# district type gets the first color, rightmost gets the last — which aids
# comparison across different words.
#
# Interpretation: this plot gives a gestalt view of the full prototypical plan.
# Where a region is strongly colored, that letter has high density there
# (consistent district boundary). Where a region is pale or grey, district
# boundaries are fuzzy or variable across the cluster's members.
#
# Note: because colors are painted sequentially and later letters overwrite
# earlier ones, overlapping regions between letters will show the color of
# the LAST letter drawn there, not a true blend. Use plot_words_centroids
# for a proper weighted-average color blend in overlap regions.
def plot_words_combined(
    gdf,
    cluster_densities,
    word,
    *,
    cmap="tab10",
    base_color="lightgrey",
    edgecolor="black",
    linewidth=0.2,
    ltr=False,
    ax=None,
    figsize=(8, 8),
):
    word = np.asarray(word, dtype=np.int32)
    cluster_densities = np.asarray(cluster_densities, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Draw the full state as a faint grey base layer; colored letters will
    # be overlaid on top of this.
    gdf.plot(
        ax=ax, color=base_color, edgecolor=edgecolor, linewidth=linewidth, alpha=0.3
    )

    if ltr and len(word) > 1:
        # Compute the density-weighted centroid x-coordinate for each letter
        # and sort letters west-to-east. This makes color assignments stable
        # and geographically interpretable across different words.
        x_centers = gdf.geometry.centroid.x.to_numpy()
        xs = []
        for centroid_id in word.tolist():
            dens = cluster_densities[int(centroid_id)]
            total = float(dens.sum())
            xs.append(
                float("inf") if total <= 0 else float(np.sum(x_centers * dens) / total)
            )
        order = np.argsort(np.asarray(xs, dtype=float), kind="mergesort")
        word = word[order]

    cmap_obj = plt.get_cmap(cmap)
    n_letters = max(len(word), 1)
    for idx, centroid_id in enumerate(word.tolist()):
        values = cluster_densities[int(centroid_id)]
        if np.all(values == 0):
            continue  # skip empty/degenerate letters
        # Assign a unique color to this letter, evenly spaced across the colormap
        color = cmap_obj(idx / max(1, n_letters - 1))
        # Build a per-precinct RGBA array: all precincts get this letter's color,
        # but with alpha = density, so high-density precincts are opaque and
        # low-density precincts are transparent (fading into the grey base).
        rgba = np.tile(color, (len(values), 1))
        if rgba.shape[1] == 3:
            rgba = np.column_stack([rgba, np.ones(len(values))])
        alpha = values / (values.max() if values.max() > 0 else 1.0)
        rgba[:, -1] = np.clip(alpha, 0.0, 1.0)
        gdf.plot(ax=ax, color=rgba, edgecolor=edgecolor, linewidth=linewidth)

    ax.axis("off")
    return fig, ax


# Visualizes a word as a single combined map using threshold-based letter
# assignment and proper weighted-average color blending for overlaps.
#
# Unlike plot_words_combined (which uses raw density as alpha), this function:
#   1. Binarizes each letter's density at `threshold` (default 0.5) — a
#      precinct is "in" a letter if its density >= 0.5, corresponding exactly
#      to the majority-vote centroid definition of a letter.
#   2. Computes a population-weighted average color for precincts claimed by
#      multiple letters (overlaps), rather than letting the last letter win.
#   3. Highlights overlap precincts with a thicker border (`overlap_edgecolor`,
#      `overlap_linewidth`) to make boundary ambiguity visually salient.
#
# Color assignment:
#   - Precincts with no letter claiming them (density < threshold for all
#     letters) are shown in `base_color` at `base_alpha` (faint grey).
#   - Precincts claimed by exactly one letter get that letter's color at
#     `centroid_alpha` (default 0.9, nearly opaque).
#   - Precincts claimed by multiple letters get a weighted-average blend of
#     those letters' colors, with weights proportional to density above the
#     threshold. These are also given a thick border to flag the overlap.
#
# The `ltr` flag works as in plot_words_combined but uses the mean x-position
# of above-threshold precincts (the actual centroid footprint) rather than
# density-weighted mean, which is slightly more robust for sparse letters.
#
# Interpretation: this is the most faithful single-map visualization of a
# word as a redistricting plan prototype. The threshold=0.5 cutoff means that
# colored regions correspond precisely to the letters (majority-vote centroids).
# Overlap regions with thick borders reveal where the letter construction
# produces ambiguous or contested precinct assignments — a direct visual
# indicator of the contiguity/coverage gaps discussed in the paper.
# Grey regions are precincts that no letter claimed, i.e. potential coverage
# gaps in the word's representation of the state.
def plot_words_centroids(
    gdf,
    cluster_densities,
    word,
    *,
    threshold=0.5,
    cmap="tab10",
    base_color="lightgrey",
    base_alpha=0.3,
    centroid_alpha=0.9,
    edgecolor="black",
    linewidth=0.2,
    overlap_edgecolor="black",
    overlap_linewidth=1.2,
    ltr=False,
    ax=None,
    figsize=(8, 8),
):
    word_arr = np.asarray(word, dtype=np.int32)
    cluster_densities = np.asarray(cluster_densities, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if ltr and len(word_arr) > 1:
        # Sort letters left-to-right by mean x-coordinate of their centroid
        # footprint (precincts with density >= threshold).
        x_centers = gdf.geometry.centroid.x.to_numpy()
        xs = []
        for centroid_id in word_arr.tolist():
            dens = cluster_densities[int(centroid_id)]
            mask = dens >= float(threshold)
            xs.append(float(np.mean(x_centers[mask])) if np.any(mask) else float("inf"))
        order = np.argsort(np.asarray(xs, dtype=float), kind="mergesort")
        word_arr = word_arr[order]

    if len(word_arr) == 0:
        gdf.plot(
            ax=ax,
            color=base_color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=base_alpha,
        )
        ax.axis("off")
        return fig, ax

    # densities: shape (n_letters, P) — density of each letter at each precinct
    densities = cluster_densities[word_arr]
    # mask: shape (n_letters, P) — True where a letter "claims" a precinct
    # (density >= threshold, i.e. the precinct is in the majority-vote centroid)
    mask = densities >= float(threshold)
    # weights: same as densities but zeroed below threshold, used for blending
    weights = densities * mask
    # totals: shape (P,) — total weight across all letters at each precinct
    totals = weights.sum(axis=0)
    # has_any: shape (P,) — True if at least one letter claims this precinct
    has_any = totals > 0
    # overlap: shape (P,) — True if MORE than one letter claims this precinct
    # These are the ambiguous boundary precincts that will get thick borders.
    overlap = mask.sum(axis=0) > 1

    cmap_obj = plt.get_cmap(cmap)
    n_letters = max(len(word_arr), 1)
    # Assign one distinct color per letter, evenly spaced across the colormap
    letter_rgba = np.array(
        [cmap_obj(i / max(1, n_letters - 1)) for i in range(len(word_arr))],
        dtype=float,
    )
    letter_rgb = letter_rgba[:, :3]  # drop alpha, handle separately

    # Compute the weighted-average RGB color at each precinct:
    # rgb_num[p] = sum over letters l of (weight[l,p] * color[l])
    # rgb[p]     = rgb_num[p] / total_weight[p]   (only where has_any[p])
    # This gives a smooth color blend in overlap regions rather than
    # letting one letter's color overwrite another's.
    rgb_num = weights.T @ letter_rgb          # shape (P, 3)
    rgb = np.zeros_like(rgb_num)
    np.divide(rgb_num, totals[:, None], out=rgb, where=has_any[:, None])

    # Start every precinct as the faint grey base color, then overwrite
    # claimed precincts with their blended letter color.
    base_rgba = np.array(mcolors.to_rgba(base_color, alpha=base_alpha), dtype=float)
    rgba = np.tile(base_rgba, (len(gdf), 1))
    rgba[has_any, :3] = rgb[has_any]
    rgba[has_any, 3] = float(centroid_alpha)

    gdf.plot(ax=ax, color=rgba, edgecolor=edgecolor, linewidth=linewidth)

    # Draw thick borders around overlap precincts to visually flag contested
    # boundary regions where two or more letters both claim the precinct —
    # these are locations where the word may not represent a valid partition.
    if np.any(overlap):
        gdf[overlap].boundary.plot(
            ax=ax, color=overlap_edgecolor, linewidth=overlap_linewidth
        )

    ax.axis("off")
    return fig, ax


# Convenience wrapper: looks up a district by its UID in the district catalog,
# decodes its precinct set from the stored string representation, and passes
# it to plot_district for rendering.
#
# `uid` indexes into df_districts.iloc, so it is a positional row index
# (not necessarily the same as district_uid if the DataFrame has been filtered
# or reindexed). Any kwargs (e.g. color, ax) are forwarded to plot_district.
def plot_district_by_uid(gdf, df_districts, uid, **kwargs):
    precincts = str_to_vec(df_districts.iloc[uid]["district_str"])
    return plot_district(gdf, precincts, **kwargs)