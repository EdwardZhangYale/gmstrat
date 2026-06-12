from shapely.geometry import MultiPoint, box
from shapely.ops import voronoi_diagram
import geopandas as gpd
import numpy as np
import json

def node_info_to_gdf(node_info, crs=None, buffer_factor=0.5):
    """
    Build a GeoDataFrame from a node_info dict (as produced by the
    generate_*_adjacency_list functions) using a Voronoi diagram to
    assign polygon geometries. Works for any planar graph layout.

    Parameters
    ----------
    node_info      : dict  {node_id: {..., x_location, y_location, ...}}
    crs            : optional CRS string to assign to the GeoDataFrame
    buffer_factor  : how much to expand the bounding box before clipping
                     Voronoi cells (prevents edge cells being cut too tight)
    """
    ids = sorted(node_info.keys())
    xs = np.array([node_info[i]["x_location"] for i in ids], dtype=float)
    ys = np.array([node_info[i]["y_location"] for i in ids], dtype=float)

    # Build a MultiPoint from all seed locations and compute Voronoi diagram.
    # envelope=True asks Shapely to return a clipped diagram, but outer cells
    # can still be large, so we clip again to a padded bounding box below.
    points = MultiPoint(list(zip(xs, ys)))
    regions = voronoi_diagram(points, envelope=points.envelope)

    # Pad the bounding box so edge cells aren't clipped too aggressively
    minx, miny, maxx, maxy = points.envelope.bounds
    dx = (maxx - minx) * buffer_factor
    dy = (maxy - miny) * buffer_factor
    bbox = box(minx - dx, miny - dy, maxx + dx, maxy + dy)
    clipped = [r.intersection(bbox) for r in regions.geoms]

    # Match each Voronoi cell back to its seed by containment.
    # For a well-separated point set this is an exact O(n²) match,
    # which is fast enough for graph sizes used in redistricting work.
    from shapely.geometry import Point
    seed_points = [Point(x, y) for x, y in zip(xs, ys)]
    ordered_polys = [None] * len(ids)
    for poly in clipped:
        for k, pt in enumerate(seed_points):
            if poly.contains(pt):
                ordered_polys[k] = poly
                break

    # Build a row per node, preserving all attributes from node_info
    rows = []
    for k, node_id in enumerate(ids):
        row = dict(node_info[node_id])
        row["geometry"] = ordered_polys[k]
        rows.append(row)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    if crs is not None:
        gdf = gdf.set_crs(crs)
    return gdf


def gdf_from_graph_json(json_path, **kwargs):
    """
    Build a GeoDataFrame from a graph JSON file as produced by
    adj_list_to_json. Passes any kwargs through to node_info_to_gdf
    (e.g. buffer_factor, crs).
    """
    with open(json_path) as f:
        data = json.load(f)

    node_info = {node["id"]: node for node in data["nodes"]}
    return node_info_to_gdf(node_info, **kwargs)