import numpy as np
import numpy.linalg as npla
from ..coptimization import lasso_skl, lasso_skli

def _alo_lasso(X, y, beta_hat, lam, thresh_tol):
    n, p = X.shape

    u = np.dot(X, beta_hat)
    XE = X[:, np.abs(beta_hat) >= thresh_tol]
    k = XE.shape[1]
    H = np.dot(XE, npla.solve(np.dot(XE.T, XE), XE.T))
    return np.diag(H) / (1.0 - np.diag(H)) * (u - y) + u

def alo_lasso(X, y, Beta_hat, lams, thresh_tol):
    n, p = X.shape
    k = lams.shape[0]
    Alo_err = np.zeros((n, k), dtype=np.float64)

    for i in range(k-1, -1, -1):
        Alo_err[:, i] = _alo_lasso(X, y, Beta_hat[:, i], lams[i], thresh_tol)

    return Alo_err

def _alo_lassoi(X, y, beta_hat, lam, thresh_tol):
    n, p = X.shape

    u = np.dot(X, beta_hat[1:]) + beta_hat[0]
    XE = X[:, np.abs(beta_hat[1:]) >= thresh_tol]
    k = XE.shape[1]
    H = np.dot(XE, npla.solve(np.dot(XE.T, XE), XE.T))
    a = n
    ldd = np.ones((n, 1), dtype=np.float64)
    H0 = H + 1.0 / (a - np.dot(ldd.T, np.dot(H, ldd))) * np.dot(ldd - np.dot(H, ldd), (ldd - np.dot(H, ldd)).T)
    return np.diag(H0) / (1.0 - np.diag(H0)) * (u - y) + u

def alo_lassoi(X, y, Beta_hat, lams, thresh_tol):
    n, p = X.shape
    k = lams.shape[0]
    Alo_err = np.zeros((n, k), dtype=np.float64)

    for i in range(k-1, -1, -1):
        Alo_err[:, i] = _alo_lassoi(X, y, Beta_hat[:, i], lams[i], thresh_tol)

    return Alo_err

