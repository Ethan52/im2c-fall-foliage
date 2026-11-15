import numpy as np
import pandas as pd

# onset from predicted and spliced brownness curves, and scores, decisions D3 and D26
ONSET_THRESHOLD = 0.6  # decision D1


def first_crossing(t: np.ndarray, v: np.ndarray, threshold: float = ONSET_THRESHOLD):
    '''first time the piecewise linear curve through the points reaches the threshold, rows are curves, NaN marks no point
    returns the time (NaN if never reached or already reached at the first point) and the mask of rows reached at the first point'''
    valid = np.isfinite(v) & np.isfinite(t)
    order = np.argsort(~valid, axis=1, kind="stable")  # valid points first, in their given time order
    vv = np.take_along_axis(np.where(valid, v, np.nan), order, axis=1)
    tt = np.take_along_axis(np.where(valid, t, np.nan), order, axis=1)
    above = vv >= threshold  # NaN compares False
    has = above.any(axis=1)
    k = above.argmax(axis=1)
    rows = np.flatnonzero(has & (k >= 1))
    k0, k1 = k[rows] - 1, k[rows]
    out = np.full(v.shape[0], np.nan)
    out[rows] = tt[rows, k0] + (threshold - vv[rows, k0]) / (vv[rows, k1] - vv[rows, k0]) * (tt[rows, k1] - tt[rows, k0])
    return out, has & (k == 0)


def spliced_onset(obs_day: np.ndarray, obs_value: np.ndarray, pred_day: np.ndarray, pred_value: np.ndarray):
    '''forecast onset of D26: observations before the issue date, then the daily prediction from the issue date on
    obs arrays have shape cell, composite with NaN where not observed or not yet available, pred_value has shape cell, day'''
    assert np.all(np.diff(pred_day) > 0)
    t = np.concatenate([obs_day, np.broadcast_to(pred_day, pred_value.shape)], axis=1)
    v = np.concatenate([obs_value, pred_value], axis=1)
    return first_crossing(t, v)


def onset_scores(forecast: np.ndarray, observed: np.ndarray):
    '''bias and mean absolute error in days over cells where both are defined, and that cell count'''
    both = np.isfinite(forecast) & np.isfinite(observed)
    err = forecast[both] - observed[both]
    return (float(err.mean()) if both.any() else np.nan, float(np.abs(err).mean()) if both.any() else np.nan, int(both.sum()))


def rmse(pred: np.ndarray, obs: np.ndarray):
    '''root mean squared error over entries where both are finite, and the count'''
    both = np.isfinite(pred) & np.isfinite(obs)
    return (float(np.sqrt(np.mean((pred[both] - obs[both]) ** 2))) if both.any() else np.nan, int(both.sum()))


def site_cells(cell_id: np.ndarray, path: str):
    '''the nearest candidate cell of each site (rank 1 of site_candidates.tsv, O10) and its position in the forest cell list'''
    s = pd.read_csv(path, sep="\t")
    s = s[s["rank"] == 1]
    pos = {r["site_id"]: int(np.searchsorted(cell_id, r["cell_id"])) for _, r in s.iterrows()}
    assert all(cell_id[p] == c for p, c in zip(pos.values(), s["cell_id"]))
    return pos
