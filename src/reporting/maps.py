import os

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRID_PATH = os.path.join(REPO_ROOT, "data", "processed", "grid", "analysis_grid.npz")
BOUNDARY_PATH = os.path.join(REPO_ROOT, "data", "raw", "maine_boundary_polygon", "Maine_State_Boundary_Polygon_Feature.shp")
SITES_CSV = os.path.join(REPO_ROOT, "configs", "sites.csv")


def load_grid():
    '''analysis grid (EPSG:26919, 1 km) with the plotting extent left, right, bottom, top in meters'''
    g = np.load(GRID_PATH)
    a, _, c, _, e, f = g["transform"]
    nrow, ncol = g["shape"]
    extent = (c, c + a * ncol, f + e * nrow, f)
    return g, extent


def to_raster(values: np.ndarray, cell_id: np.ndarray, grid):
    '''values on a list of cells placed on the analysis raster, NaN elsewhere'''
    pos = np.searchsorted(grid["cell_id"], cell_id)
    assert np.array_equal(grid["cell_id"][pos], cell_id), "cells missing from the analysis grid"
    out = np.full(tuple(grid["shape"]), np.nan)
    out[grid["row"][pos], grid["col"][pos]] = values
    return out


def load_boundary(epsg: int):
    '''Maine state outline in the analysis CRS'''
    return gpd.read_file(BOUNDARY_PATH).to_crs(epsg=epsg)


def underlay(cell_id: np.ndarray, valid: np.ndarray, grid):
    '''raster equal to one on the analysis cells (forest cells with valid U and L), NaN elsewhere'''
    return to_raster(np.where(valid, 1.0, np.nan), cell_id, grid)


def map_panels(axes, rasters: list, titles: list, extent: tuple, boundary, under: np.ndarray, cmap, vmin: float, vmax: float, sites=None):
    '''one statewide map per axis with a common color scale; analysis cells without a value light gray, other land white'''
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad((0, 0, 0, 0))
    gray = plt.get_cmap("Greys").copy()
    gray.set_bad((0, 0, 0, 0))
    im = None
    for ax, r, title in zip(axes, rasters, titles):
        ax.imshow(under, extent=extent, cmap=gray, vmin=0, vmax=4, interpolation="nearest")
        im = ax.imshow(r, extent=extent, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
        boundary.boundary.plot(ax=ax, color="k", linewidth=0.4)
        if sites is not None:
            ax.plot(sites["easting_m"], sites["northing_m"], "k^", ms=4)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    return im
