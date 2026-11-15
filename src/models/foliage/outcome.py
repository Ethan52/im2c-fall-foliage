import logging

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

# outcome models of manuscript section 3.4, decision D25
# linear regression (eq. 7 to 10) and a GAM (eq. 11 to 18) with mgcv's penalized cubic regression spline (bs="cr"),
# sum-to-zero constraints absorbed by QR and mgcv's penalty scaling, one shared lambda chosen by GCV
N_KNOTS = 10  # mgcv default basis dimension
LOG10_LAMBDA_RANGE = (-8.0, 8.0)
CHUNK = 1_000_000  # rows per design matrix chunk


def cr_knots(x: np.ndarray, k: int):
    '''knots spread evenly through the sorted unique values, linear interpolation between them, as mgcv place.knots'''
    u = np.unique(x)
    assert len(u) >= k, "fewer unique values than knots"
    pos = np.arange(k) * (len(u) - 1) / (k - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int64), len(u) - 2)
    frac = pos - i0
    return u[i0] * (1 - frac) + u[i0 + 1] * frac


def cr_matrices(knots: np.ndarray):
    '''map from knot values to knot second derivatives of the natural cubic spline, and the penalty matrix of the integrated squared second derivative
    following Wood (2017), Generalized Additive Models, section 5.3.1'''
    k = len(knots)
    h = np.diff(knots)
    D = np.zeros((k - 2, k))
    B = np.zeros((k - 2, k - 2))
    for i in range(k - 2):
        D[i, i] = 1.0 / h[i]
        D[i, i + 1] = -1.0 / h[i] - 1.0 / h[i + 1]
        D[i, i + 2] = 1.0 / h[i + 1]
        B[i, i] = (h[i] + h[i + 1]) / 3.0
        if i < k - 3:
            B[i, i + 1] = B[i + 1, i] = h[i + 1] / 6.0
    BinvD = np.linalg.solve(B, D)
    F = np.zeros((k, k))
    F[1:-1] = BinvD  # second derivatives are zero at the end knots of a natural spline
    S = D.T @ BinvD
    return F, 0.5 * (S + S.T)


def cr_basis(x: np.ndarray, knots: np.ndarray, F: np.ndarray):
    '''basis matrix of shape row, knot: the natural cubic spline with the given knot values, linear beyond the end knots'''
    k = len(knots)
    h = np.diff(knots)
    X = np.zeros((len(x), k))
    inside = (x >= knots[0]) & (x <= knots[-1])
    r = np.flatnonzero(inside)
    j = np.clip(np.searchsorted(knots, x[r], side="right") - 1, 0, k - 2)
    hj = h[j]
    am = (knots[j + 1] - x[r]) / hj
    ap = (x[r] - knots[j]) / hj
    cm = ((knots[j + 1] - x[r]) ** 3 / hj - hj * (knots[j + 1] - x[r])) / 6.0
    cp = ((x[r] - knots[j]) ** 3 / hj - hj * (x[r] - knots[j])) / 6.0
    X[r, j] += am
    X[r, j + 1] += ap
    X[r] += cm[:, None] * F[j] + cp[:, None] * F[j + 1]
    # linear extrapolation with the end slopes of the spline
    e = np.eye(k)
    lo = np.flatnonzero(x < knots[0])
    if len(lo):
        slope = (e[1] - e[0]) / h[0] - h[0] * (2 * F[0] + F[1]) / 6.0
        X[lo] = e[0] + (x[lo] - knots[0])[:, None] * slope
    hi = np.flatnonzero(x > knots[-1])
    if len(hi):
        slope = (e[k - 1] - e[k - 2]) / h[k - 2] + h[k - 2] * (F[k - 2] + 2 * F[k - 1]) / 6.0
        X[hi] = e[k - 1] + (x[hi] - knots[-1])[:, None] * slope
    return X


def setup_smooths(X: np.ndarray):
    '''knots, constraint and scaled penalty of one smooth per covariate column, from the training covariates of shape row, covariate'''
    smooths = []
    for c in range(X.shape[1]):
        x = X[:, c]
        knots = cr_knots(x, N_KNOTS)
        F, S = cr_matrices(knots)
        colsum = np.zeros(N_KNOTS)
        for s in range(0, len(x), CHUNK):
            colsum += cr_basis(x[s:s + CHUNK], knots, F).sum(axis=0)
        # sum-to-zero constraint over the training rows, absorbed by QR as in mgcv
        q, _ = np.linalg.qr(colsum[:, None], mode="complete")
        Z = q[:, 1:]  # knot, constrained coefficient
        maxx = 0.0
        for s in range(0, len(x), CHUNK):
            maxx = max(maxx, np.abs(cr_basis(x[s:s + CHUNK], knots, F)).sum(axis=1).max())
        # mgcv smoothCon penalty scaling before the constraint: the penalty is divided by its one-norm and multiplied by the squared infinity-norm of the model matrix
        S = S * maxx ** 2 / np.abs(S).sum(axis=0).max()
        smooths.append({"knots": knots, "F": F, "Z": Z, "S": Z.T @ S @ Z})
    return smooths


def gam_design(X: np.ndarray, smooths: list):
    '''GAM design matrix of shape row, coefficient: intercept, then the constrained basis of each smooth (eq. 13)'''
    cols = [np.ones((X.shape[0], 1))]
    for c, sm in enumerate(smooths):
        cols.append(cr_basis(X[:, c], sm["knots"], sm["F"]) @ sm["Z"])
    return np.hstack(cols)


def lr_design(X: np.ndarray):
    '''linear regression design matrix: intercept, then the covariates (eq. 8)'''
    return np.hstack([np.ones((X.shape[0], 1)), X])


def penalty(smooths: list):
    '''block diagonal penalty of the GAM, zero for the intercept'''
    sizes = [1] + [sm["S"].shape[0] for sm in smooths]
    P = np.zeros((sum(sizes), sum(sizes)))
    start = 1
    for sm in smooths:
        n = sm["S"].shape[0]
        P[start:start + n, start:start + n] = sm["S"]
        start += n
    return P


def gram(X: np.ndarray, y: np.ndarray, design, *args):
    '''sums X'X, X'y, y'y and row count of a design built in chunks'''
    G = None
    for s in range(0, len(y), CHUNK):
        A = design(X[s:s + CHUNK], *args)
        if G is None:
            G = np.zeros((A.shape[1], A.shape[1]))
            b = np.zeros(A.shape[1])
        G += A.T @ A
        b += A.T @ y[s:s + CHUNK]
    return G, b, float(y @ y), len(y)


def penalized_fit(G: np.ndarray, b: np.ndarray, yy: float, n: int, P: np.ndarray, lam: float):
    '''solution of eq. 18, residual sum of squares, effective degrees of freedom and GCV score (Craven and Wahba 1979)'''
    A = G + lam * P
    beta = np.linalg.solve(A, b)
    rss = yy - 2 * beta @ b + beta @ G @ beta
    edf = float(np.trace(np.linalg.solve(A, G)))
    return beta, rss, edf, n * rss / (n - edf) ** 2


def choose_lambda(G: np.ndarray, b: np.ndarray, yy: float, n: int, P: np.ndarray):
    '''lambda minimizing GCV: a grid in log10 lambda, then a bounded scalar search around the best grid point'''
    grid = np.arange(LOG10_LAMBDA_RANGE[0], LOG10_LAMBDA_RANGE[1] + 1e-9, 0.25)
    scores = np.array([penalized_fit(G, b, yy, n, P, 10.0 ** g)[3] for g in grid])
    i = int(np.argmin(scores))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda g: penalized_fit(G, b, yy, n, P, 10.0 ** g)[3], bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-3})
    best = res.x if res.fun <= scores[i] else grid[i]
    if i in (0, len(grid) - 1):
        logger.info("GCV minimum at the edge of the lambda grid, log10 lambda %.2f", grid[i])
    return 10.0 ** best, grid, scores


def predict(X: np.ndarray, beta: np.ndarray, smooths: list):
    '''predicted brownness at covariates of shape row, covariate; linear regression when smooths is None'''
    out = np.empty((X.shape[0],) + beta.shape[1:])  # row, or row and coefficient set when beta has two dimensions
    for s in range(0, X.shape[0], CHUNK):
        A = lr_design(X[s:s + CHUNK]) if smooths is None else gam_design(X[s:s + CHUNK], smooths)
        out[s:s + CHUNK] = A @ beta
    return out


def partial_effect(x: np.ndarray, c: int, beta: np.ndarray, smooths: list):
    '''the smooth of covariate c evaluated at x, under the sum-to-zero constraint'''
    start = 1 + sum(sm["Z"].shape[1] for sm in smooths[:c])
    sm = smooths[c]
    return cr_basis(x, sm["knots"], sm["F"]) @ sm["Z"] @ beta[start:start + sm["Z"].shape[1]]


def save_smooths(smooths: list):
    '''stacked arrays of the smooth definitions, for np.savez'''
    return {"knots": np.stack([s["knots"] for s in smooths]), "F": np.stack([s["F"] for s in smooths]),
            "Z": np.stack([s["Z"] for s in smooths]), "S": np.stack([s["S"] for s in smooths])}


def load_smooths(d):
    '''smooth definitions from a loaded fit file'''
    return [{"knots": d["knots"][c], "F": d["F"][c], "Z": d["Z"][c], "S": d["S"][c]} for c in range(d["knots"].shape[0])]


def load_fit(path: str):
    '''fitted coefficients, smooths, noise variances and prior precisions of both models'''
    d = np.load(path)
    return {"lr": {"beta": d["lr_beta"], "smooths": None, "gamma": float(d["lr_gamma"]), "Q": d["lr_Q"]},
            "gam": {"beta": d["gam_beta"], "smooths": load_smooths({k: d[f"gam_{k}"] for k in ("knots", "F", "Z", "S")}),
                    "gamma": float(d["gam_gamma"]), "Q": d["gam_Q"]}}
