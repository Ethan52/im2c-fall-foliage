import os
import sys
import logging

import numpy as np
import pandas as pd
import pyproj
import rasterio
from scipy.io import netcdf_file

logger = logging.getLogger(__name__)

# checks the downloaded raw data after the download jobs finish, see Verification in references/project_decisions.md
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(REPO_ROOT, "data", "raw")
SITES_CSV = os.path.join(REPO_ROOT, "configs", "sites.csv")
DAYMET_CRS = "+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
YEAR = 2025


def record_status(name: str):
    '''log the status counts of one download record'''
    record = pd.read_csv(os.path.join(RAW, name, "download_record.tsv"), sep="\t")
    logger.info("%s record: %s", name, record["status"].value_counts().to_dict())
    return record


def daymet_site_series(var: str, sites: pd.DataFrame):
    '''daily Daymet values of one variable in YEAR at the cell nearest each site; return days by sites'''
    with netcdf_file(os.path.join(RAW, "daymet", f"daymet_v4_daily_maine_{var}_{YEAR}.nc"), "r", mmap=False) as nc:
        x = nc.variables["x"].data.copy()
        y = nc.variables["y"].data.copy()
        data = nc.variables[var].data  # time, y, x
        sx, sy = pyproj.Transformer.from_crs(4326, DAYMET_CRS, always_xy=True).transform(sites["longitude"].values, sites["latitude"].values)
        ix = np.abs(x[None, :] - sx[:, None]).argmin(axis=1)
        iy = np.abs(y[None, :] - sy[:, None]).argmin(axis=1)
        series = data[:, iy, ix].astype(np.float64)  # day, site
    assert np.all(series != -9999.0), "a site falls on a Daymet fill cell"
    return series


def gridmet_site_series(prefix: str, varname: str, sites: pd.DataFrame):
    '''daily decoded gridMET values in YEAR at the cell nearest each site; return days by sites'''
    with netcdf_file(os.path.join(RAW, "gridmet", f"gridmet_maine_{prefix}_{YEAR}.nc"), "r", mmap=False) as nc:
        var = nc.variables[varname]
        lat = nc.variables["lat"].data.copy()
        lon = nc.variables["lon"].data.copy()
        ilat = np.abs(lat[None, :] - sites["latitude"].values[:, None]).argmin(axis=1)
        ilon = np.abs(lon[None, :] - sites["longitude"].values[:, None]).argmin(axis=1)
        raw = var.data.view(">u2")[:, ilat, ilon]  # day, site
        assert np.all(raw != 32767), "a site falls on a gridMET fill cell"
        return raw.astype(np.float64) * float(var.scale_factor) + float(var.add_offset)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    sites = pd.read_csv(SITES_CSV)
    names = sites["site_name"].tolist()

    for name in ("daymet", "gridmet", "nlcd", "dem", "modis_2026"):
        if not os.path.exists(os.path.join(RAW, name, "download_record.tsv")):
            logger.warning("%s has no download record yet, skipped", name)
            continue
        record_status(name)

    # Daymet and gridMET at the sites, same year
    tmax_d = daymet_site_series("tmax", sites)  # degrees C
    tmax_g = gridmet_site_series("tmmx", "air_temperature", sites) - 273.15  # K to degrees C
    logger.info("Daymet days %d, gridMET days %d in %d", tmax_d.shape[0], tmax_g.shape[0], YEAR)
    jan_d = tmax_d[:31].mean(axis=0)
    jan_g = tmax_g[:31].mean(axis=0)
    for i, name in enumerate(names):
        logger.info("%s January %d mean tmax: Daymet %.2f C, gridMET %.2f C", name, YEAR, jan_d[i], jan_g[i])
    # Daymet drops December 31 only in leap years, so in 2025 the first 365 days of both line up
    n = min(tmax_d.shape[0], tmax_g.shape[0])
    for i, name in enumerate(names):
        r = np.corrcoef(tmax_d[:n, i], tmax_g[:n, i])[0, 1]
        bias = np.mean(tmax_g[:n, i] - tmax_d[:n, i])
        logger.info("%s daily tmax %d: correlation Daymet and gridMET %.4f, mean gridMET minus Daymet %.2f C", name, YEAR, r, bias)

    # NLCD 2021 classes in the 1 km window around each site
    with rasterio.open(os.path.join(RAW, "nlcd", "nlcd_2021_land_cover_maine.tif")) as src:
        logger.info("NLCD CRS %s, resolution %s, size %d by %d, dtype %s", src.crs, src.res, src.width, src.height, src.dtypes[0])
        sx, sy = pyproj.Transformer.from_crs(4326, 5070, always_xy=True).transform(sites["longitude"].values, sites["latitude"].values)
        for i, name in enumerate(names):
            row, col = src.index(sx[i], sy[i])
            window = src.read(1, window=((row - 16, row + 17), (col - 16, col + 17)))  # about 1 km by 1 km
            frac = {c: float((window == c).mean()) for c in (41, 42, 43)}
            logger.info("%s NLCD 2021: center class %d, 1 km window fraction deciduous %.2f, evergreen %.2f, mixed %.2f",
                        name, window[16, 16], frac[41], frac[42], frac[43])

    # NASADEM elevation layer; the source-number layer AppEEARS adds is not elevation and is skipped
    dem_files = sorted(f for root, _, files in os.walk(os.path.join(RAW, "dem")) for f in [os.path.join(root, g) for g in files] if f.endswith(".tif") and "NASADEM_HGT" in f)
    for path in dem_files:
        with rasterio.open(path) as src:
            # statistics computed by GDAL without loading the full raster, nodata excluded
            stats = src.stats(indexes=[1], approx=False)[0]
            logger.info("DEM %s: CRS %s, resolution %s, size %d by %d, dtype %s, nodata %s, valid %.0f to %.0f m",
                        os.path.basename(path), src.crs, src.res, src.width, src.height, src.dtypes[0], src.nodata, stats.min, stats.max)
            elevations = [int(v[0]) for v in src.sample(zip(sites["longitude"].values, sites["latitude"].values))]
            for name, h in zip(names, elevations):
                logger.info("%s elevation %d m", name, h)

    # MODIS 2026 grid against the 2001 to 2025 download
    modis_dir = os.path.join(RAW, "modis", "MOD09A1.061_2000360_to_2025365")
    ref = sorted(f for f in os.listdir(modis_dir) if "sur_refl_b01" in f)[-1]
    new_files = sorted(f for root, _, files in os.walk(os.path.join(RAW, "modis_2026")) for f in [os.path.join(root, g) for g in files] if f.endswith(".tif"))
    b01 = [f for f in new_files if "sur_refl_b01" in f]
    if not b01:
        logger.warning("no MODIS 2026 GeoTIFFs yet, grid check skipped")
        return
    logger.info("MODIS 2026: %d GeoTIFFs, %d red band composites, first %s, last %s",
                len(new_files), len(b01), os.path.basename(b01[0]) if b01 else None, os.path.basename(b01[-1]) if b01 else None)
    with rasterio.open(os.path.join(modis_dir, ref)) as a, rasterio.open(b01[0]) as b:
        same = a.crs == b.crs and a.transform == b.transform and a.shape == b.shape and a.nodata == b.nodata and a.dtypes == b.dtypes
        logger.info("MODIS 2026 grid equals the 2001 to 2025 grid: %s (shape %s, transform %s)", same, b.shape, tuple(b.transform)[:6])


if __name__ == "__main__":
    main()
