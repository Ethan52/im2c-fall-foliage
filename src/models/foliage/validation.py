import sys
import logging

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from models.foliage import features, outcome, assimilation, evaluation

logger = logging.getLogger(__name__)

# runnable checks of the foliage modules on synthetic data, no project data needed
SEED = 0
TOL = 1e-8


def check_spline():
    '''basis against scipy's natural cubic spline, linear extrapolation, and the penalty against quadrature'''
    rng = np.random.default_rng(SEED)
    knots = np.sort(rng.uniform(0, 10, outcome.N_KNOTS))
    beta = rng.normal(size=outcome.N_KNOTS)
    F, S = outcome.cr_matrices(knots)
    cs = CubicSpline(knots, beta, bc_type="natural")
    x = rng.uniform(knots[0], knots[-1], 1000)
    err_in = np.abs(outcome.cr_basis(x, knots, F) @ beta - cs(x)).max()
    xo = np.array([knots[0] - 2.0, knots[0] - 0.1, knots[-1] + 0.1, knots[-1] + 3.0])
    lin = np.where(xo < knots[0], beta[0] + (xo - knots[0]) * cs(knots[0], 1), beta[-1] + (xo - knots[-1]) * cs(knots[-1], 1))
    err_out = np.abs(outcome.cr_basis(xo, knots, F) @ beta - lin).max()
    pen = sum(quad(lambda z: cs(z, 2) ** 2, knots[i], knots[i + 1])[0] for i in range(len(knots) - 1))
    err_pen = abs(beta @ S @ beta - pen) / pen
    logger.info("spline: max error inside %.2e, outside %.2e, relative penalty error %.2e", err_in, err_out, err_pen)
    assert err_in < 1e-10 and err_out < 1e-10 and err_pen < 1e-8


def check_gam():
    '''recovery of known coefficients, constraint, and GCV from sums against the explicit hat matrix'''
    rng = np.random.default_rng(SEED)
    n = 4000
    X = rng.uniform(0, 1, (n, 2))
    smooths = outcome.setup_smooths(X)
    A = outcome.gam_design(X, smooths)
    beta_true = rng.normal(size=A.shape[1])
    y = A @ beta_true + 0.01 * rng.normal(size=n)
    G, b, yy, nn = outcome.gram(X, y, outcome.gam_design, smooths)
    P = outcome.penalty(smooths)
    beta, rss, edf, gcv = outcome.penalized_fit(G, b, yy, nn, P, 1e-10)
    err_beta = np.abs(beta - beta_true).max()
    constraint = max(abs(A[:, 1 + 9 * c: 10 + 9 * c].sum(axis=0)).max() for c in range(2))
    lam = 1e-2
    beta, rss, edf, gcv = outcome.penalized_fit(G, b, yy, nn, P, lam)
    H = A @ np.linalg.solve(A.T @ A + lam * P, A.T)
    rss_direct = float(((y - H @ y) ** 2).sum())
    gcv_direct = n * rss_direct / (n - np.trace(H)) ** 2
    logger.info("GAM: max coefficient error %.2e, max constrained column sum %.2e, GCV from sums %.10e, direct %.10e, edf %.4f against %.4f",
                err_beta, constraint, gcv, gcv_direct, edf, np.trace(H))
    assert err_beta < 1e-2 and constraint < 1e-6 and abs(gcv - gcv_direct) / gcv_direct < 1e-6 and abs(edf - np.trace(H)) < 1e-6


def check_kalman():
    '''with no process noise, sequential updates, the batch update and the update from sums equal the fit on all data'''
    rng = np.random.default_rng(SEED)
    p, na, nb = 6, 300, 40
    Pen = np.diag([0.0] + [1.0] * (p - 1))
    lam, gamma = 0.5, 0.2
    Xa, Xb = rng.normal(size=(na, p)), rng.normal(size=(nb, p))
    beta = rng.normal(size=p)
    ya, yb = Xa @ beta + rng.normal(size=na), Xb @ beta + rng.normal(size=nb)
    Q0 = (Xa.T @ Xa + lam * Pen) / gamma
    m0 = np.linalg.solve(Xa.T @ Xa + lam * Pen, Xa.T @ ya)
    C0 = np.linalg.inv(Q0)
    m_seq, C_seq = assimilation.update_sequential(m0, C0, Xb, yb, gamma)
    m_bat, C_bat = assimilation.update_batch(m0, C0, Xb, yb, gamma)
    m_sum, Q_sum = assimilation.update_from_sums(m0, Q0, Xb.T @ Xb, Xb.T @ yb, gamma)
    m_all = np.linalg.solve(Xa.T @ Xa + Xb.T @ Xb + lam * Pen, Xa.T @ ya + Xb.T @ yb)
    errs = [np.abs(m_seq - m_all).max(), np.abs(m_bat - m_all).max(), np.abs(m_sum - m_all).max(),
            np.abs(C_seq - np.linalg.inv(Q_sum)).max()]
    logger.info("Kalman: differences from the fit on all data, sequential %.2e, batch %.2e, sums %.2e; covariance %.2e", *errs)
    assert max(errs) < 1e-8


def check_weather():
    '''weather covariates against a direct loop at one cell'''
    rng = np.random.default_rng(SEED)
    dates = np.arange(np.datetime64("2010-06-01"), np.datetime64("2010-12-01"))
    n = len(dates)
    tmax = rng.uniform(5, 30, (2, n))
    tmin = tmax - rng.uniform(2, 12, (2, n))
    prcp = rng.exponential(3, (2, n))
    srad = rng.uniform(50, 400, (2, n))
    vp = rng.uniform(300, 2000, (2, n))
    dayl = rng.uniform(30000, 55000, (2, n))
    w = features.weather_covariates(tmax, tmin, prcp, srad, vp, dayl, dates)
    d = int(np.flatnonzero(dates == np.datetime64("2010-09-15"))[0])
    tm = 0.5 * (tmax[1] + tmin[1])
    a = np.flatnonzero(dates == np.datetime64("2010-08-01"))[0]
    vpd = features.saturation_vp(tm) - vp[1]
    gsi = (np.clip((tmin[1] + 2) / 7, 0, 1) * (1 - np.clip((vpd - 900) / 3200, 0, 1)) * np.clip(dayl[1] / 3600 - 10, 0, 1))
    hand = {"T60": tm[d - 59:d + 1].mean(), "P60": prcp[1, d - 59:d + 1].sum(), "R30": (srad[1] * dayl[1] / 1e6)[d - 29:d + 1].mean(),
            "CDD": np.maximum(0, 20 - tm[a:d + 1]).sum(), "GSI": gsi[d - 20:d + 1].mean()}
    errs = {k: abs(w[k][1, d] - v) for k, v in hand.items()}
    logger.info("weather covariates: differences from a direct computation %s", {k: f"{e:.1e}" for k, e in errs.items()})
    assert max(errs.values()) < 1e-8
    assert np.isnan(w["T60"][1, 58]) and np.isfinite(w["T60"][1, 59]) and np.isnan(w["CDD"][1, a - 1])


def check_crossing():
    '''first crossing on a small hand example'''
    t = np.array([[1.0, 2.0, 3.0, np.nan], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    v = np.array([[0.2, 0.5, 0.7, np.nan], [0.7, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 0.4]])
    onset, first = evaluation.first_crossing(t, v)
    logger.info("first crossing: %s, reached at first point %s", onset, first)
    assert abs(onset[0] - 2.5) < TOL and np.isnan(onset[1]) and first[1] and np.isnan(onset[2]) and not first[2]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    check_spline()
    check_gam()
    check_kalman()
    check_weather()
    check_crossing()
    logger.info("all foliage checks passed")


if __name__ == "__main__":
    main()
