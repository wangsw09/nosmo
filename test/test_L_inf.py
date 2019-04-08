import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import alo

TOL = 1e-4
def test_L_inf():
    abs_tol = 1e-12
    n = 3
    p = 2
    X = np.arange(6, dtype=np.float64).reshape((3, 2))
    y = np.array([0, 1, 3], dtype=np.float64)
    lam = 2
    beta_pxgd = alo.coptimization.L_inf_pxgd(X, y, np.zeros(2, dtype=np.float64), lam, abs_tol, 1000)
    beta_arpxgd = alo.coptimization.L_inf_arpxgd(X, y, np.zeros(2, dtype=np.float64), lam, abs_tol, 1000)
    zu_admm = alo.coptimization.L_inf_admm(X, y, lam, 5000, abs_tol, 100000)
    beta0_true = np.array([0.28037383, 0.28037383], dtype=np.float64)
    assert np.abs(beta_pxgd - beta0_true).max() < TOL
    assert np.abs(beta_arpxgd - beta0_true).max() < TOL
    assert np.abs(zu_admm[ : p] - np.dot(X.T, y - np.dot(X, beta0_true))).max() < TOL
    assert np.abs(zu_admm[p : n + p] - (y - np.dot(X, beta0_true))).max() < TOL
    lam = 0.3
    beta_pxgd = alo.coptimization.L_inf_pxgd(X, y, np.zeros(2, dtype=np.float64), lam, abs_tol, 1000)
    beta_arpxgd = alo.coptimization.L_inf_arpxgd(X, y, np.zeros(2, dtype=np.float64), lam, abs_tol, 100000)
    zu_admm = alo.coptimization.L_inf_admm(X, y, lam, 5000, abs_tol, 100000)
    beta0_true = np.array([0.47910847, 0.15837712], dtype=np.float64)
    assert np.abs(beta_pxgd - beta0_true).max() < TOL
    assert np.abs(beta_arpxgd - beta0_true).max() < TOL
    assert np.abs(zu_admm[ : p] - np.dot(X.T, y - np.dot(X, beta0_true))).max() < TOL
    assert np.abs(zu_admm[p : n + p] - (y - np.dot(X, beta0_true))).max() < TOL
    
