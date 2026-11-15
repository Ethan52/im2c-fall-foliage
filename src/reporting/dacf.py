import os

import numpy as np
import pandas as pd
from pyproj import Transformer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DACF_DIR = os.path.join(REPO_ROOT, "data", "raw", "maine_foliage_reports")

# DACF color-change categories in order, with their percent ranges read as brownness fractions (assumption, category maps only)
CATEGORIES = ["Very Low", "Low", "Moderate", "High", "Peak", "Past Peak"]
BROWNNESS_EDGES = [0.1, 0.3, 0.5, 0.7]  # Very Low below 0.1, Low 0.1 to 0.3, Moderate 0.3 to 0.5, High 0.5 to 0.7, Peak at least 0.7
# sub-zones used by any report; a report row for a whole zone applies to both of its halves
SUBZONES = ["1N", "1S", "2", "3N", "3S", "4N", "4S", "5N", "5S", "6", "7"]


def load_tables():
    '''hand-transcribed weekly zone categories (2024 to 2026), archive peak weeks (2020 to 2024) and zone towns'''
    cat = pd.read_csv(os.path.join(DACF_DIR, "zone_categories.tsv"), sep="\t", dtype={"zone": str}, keep_default_na=False)
    cat["report_date"] = pd.to_datetime(cat["report_date"])
    peaks = pd.read_csv(os.path.join(DACF_DIR, "peak_weeks_archive.tsv"), sep="\t", dtype={"zone": str})
    peaks["peak_report_date"] = pd.to_datetime(peaks["peak_report_date"])
    towns = pd.read_csv(os.path.join(DACF_DIR, "zone_towns.tsv"), sep="\t")
    assert set(cat["color_change"]) <= set(CATEGORIES), f"unknown categories {set(cat['color_change']) - set(CATEGORIES)}"
    return cat, peaks, towns


def expand_subzones(table: pd.DataFrame, zone_col: str = "zone"):
    '''one row per sub-zone: a whole-zone row (for example 4) is copied to its halves (4N and 4S)'''
    rows = []
    for _, r in table.iterrows():
        z = r[zone_col]
        for s in [s for s in SUBZONES if s == z or s.rstrip("NS") == z]:
            rows.append({**r.to_dict(), zone_col: s})
    return pd.DataFrame(rows)


def cell_zones(x: np.ndarray, y: np.ndarray, towns: pd.DataFrame, epsg: int):
    '''sub-zone of the nearest named town for each cell center (meters in the analysis CRS)'''
    tx, ty = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(towns["longitude"].values, towns["latitude"].values)
    d2 = (x[:, None] - tx[None, :]) ** 2 + (y[:, None] - ty[None, :]) ** 2  # cell, town
    return towns["zone"].values[np.argmin(d2, axis=1)]


def report_peak_dates(cat: pd.DataFrame):
    '''first report date per year and sub-zone with color change Peak, or Past Peak if Peak was never reported'''
    sub = expand_subzones(cat)
    sub["year"] = sub["report_date"].dt.year
    out = []
    for (year, zone), g in sub.groupby(["year", "zone"]):
        hit = g[g["color_change"].isin(["Peak", "Past Peak"])].sort_values("report_date")
        out.append({"year": year, "zone": zone, "peak_report_date": hit["report_date"].iloc[0] if len(hit) else pd.NaT,
                    "first_label": hit["color_change"].iloc[0] if len(hit) else ""})
    return pd.DataFrame(out)


def brownness_category(b: np.ndarray):
    '''DACF category index 0 to 4 (Very Low to Peak) from brownness, NaN kept; Past Peak is not distinguished from brownness'''
    k = np.digitize(b, BROWNNESS_EDGES).astype(float)
    k[~np.isfinite(b)] = np.nan
    return k
