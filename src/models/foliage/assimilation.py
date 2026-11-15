import numpy as np

# Kalman filter on the coefficients of F, manuscript section 3.5, decision D26
# signal model with A equal to the identity and process noise Sigma, data model u = phi' beta + noise of variance Gamma


def predict_step(m: np.ndarray, C: np.ndarray, Sigma: np.ndarray):
    '''prediction equations 1 and 2 of section 3.5.2 with A equal to the identity'''
    return m.copy(), C + Sigma


def update_sequential(m: np.ndarray, C: np.ndarray, Phi: np.ndarray, u: np.ndarray, gamma: float):
    '''update equations 3 to 6 of section 3.5.2 applied once per scalar observation, rows of Phi are feature vectors'''
    m = m.copy()
    C = C.copy()
    for phi, obs in zip(Phi, u):
        S = phi @ C @ phi + gamma  # innovation variance
        K = C @ phi / S  # Kalman gain
        m = m + K * (obs - phi @ m)
        C = C - np.outer(K, phi @ C)
    return m, 0.5 * (C + C.T)


def update_batch(m: np.ndarray, C: np.ndarray, Phi: np.ndarray, u: np.ndarray, gamma: float):
    '''the same update for a batch of observations with no process noise between them, in information form'''
    Cinv = np.linalg.inv(C)
    P = Cinv + Phi.T @ Phi / gamma
    C_new = np.linalg.inv(P)
    m_new = C_new @ (Cinv @ m + Phi.T @ u / gamma)
    return m_new, 0.5 * (C_new + C_new.T)


def update_from_sums(m: np.ndarray, Q: np.ndarray, G: np.ndarray, b: np.ndarray, gamma: float):
    '''the batch update in precision form, Q the inverse of C, given the sums Phi'Phi and Phi'u of the new observations
    avoids inverting the very small training posterior covariance'''
    Q_new = Q + G / gamma
    m_new = np.linalg.solve(Q_new, Q @ m + b / gamma)
    return m_new, 0.5 * (Q_new + Q_new.T)
