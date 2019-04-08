import numpy as np
import numpy.linalg as npla
from ..coptimization import svm_skl, svm_skli

def _alo_svm(X, y, beta_hat, lam, thresh_tol):
    n, p = X.shape

    u = np.dot(X, beta_hat) * y
    V = (np.abs(u - 1.0) < thresh_tol)
    S = np.logical_not(V)
    
    a = np.zeros(n, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)
    
    if np.any(V):
        Y = npla.inv(np.dot(X[V, :], X[V, :].T))
        
        a[S] = np.diag(np.dot(np.dot(X[S, :], np.eye(p) - 
            np.dot(np.dot(X[V, :].T, Y), X[V, :])), X[S, :].T)) / lam
        a[V] = 1.0 / lam / np.diag(Y)
        
        g[S] = np.where(u[S] > 1, 0, -y[S])
        g[V] = npla.lstsq(X[V, :].T, - np.dot(X[S, :].T, g[S]) - lam * beta_hat)[0]
    else:
        a = np.diag(np.dot(X, X.T)) / lam
        g = np.where(u > 1, 0, -y[S])
        
    return np.dot(X, beta_hat) + a * g
 
def alo_svm(X, y, Beta_hat, lams, thresh_tol):
    n, p = X.shape
    k = lams.shape[0]
    Alo_err = np.zeros((n, k), dtype=np.float64)
    for i in range(k-1, -1, -1):
        Alo_err[:, i] = _alo_svm(X, y, Beta_hat[:, i], lams[i], thresh_tol)
    return Alo_err

def _alo_svmi(X, y, beta_hat, lam, thresh_tol):
    n, p = X.shape

    u = (np.dot(X, beta_hat[1:]) + beta_hat[0]) * y
    V = (np.abs(u - 1.0) < thresh_tol)
    S = np.logical_not(V)
    
    a = np.zeros(n, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)

    if np.any(V):
        Y = npla.inv(np.dot(X[V, :], X[V, :].T))
        Yoneu = np.dot(Y, np.ones(Y.shape[0], dtype=np.float64))
        U = Y - np.dot(Yoneu[:, np.newaxis], Yoneu[np.newaxis, :]) / np.sum(Y)
        d = np.dot(np.dot(X[S, :], X[V, :].T), Yoneu) - 1.0
        W = (np.dot(np.dot(X[S, :], np.eye(p) - np.dot(np.dot(X[V, :].T, Y),
            X[V, :])), X[S, :].T) + np.dot(d[:, np.newaxis],
                d[np.newaxis, :]) / np.sum(Y)) / lam
        
        a[S] = np.diag(W)
        a[V] = 1.0 / lam / np.diag(U)
        
        g[S] = np.where(u[S] > 1, 0, -y[S])
        g[V] = npla.lstsq(X[V, :].T, - np.dot(X[S, :].T, g[S]) - lam * beta_hat[1:])[0]
    else:
        a = np.diag(np.dot(X, X.T)) / lam / (1 - 1.0 / n)
        g = np.where(u > 1, 0, -y[S])
        
    return np.dot(X, beta_hat[1:]) + beta_hat[0] + a * g
 
def alo_svmi(X, y, Beta_hat, lams, thresh_tol):
    n, p = X.shape
    k = lams.shape[0]
    Alo_err = np.zeros((n, k), dtype=np.float64)
    for i in range(k-1, -1, -1):
        Alo_err[:, i] = _alo_svmi(X, y, Beta_hat[:, i], lams[i], thresh_tol)
    return Alo_err

