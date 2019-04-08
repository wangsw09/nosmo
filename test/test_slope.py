import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import alo

TOL = 1e-7
def test_slope():
    abs_tol = 1e-12
    X = np.arange(15, dtype=np.float64).reshape((5, 3))
    y = np.array([0, 5, 12, -4, -8], dtype=np.float64)
    theta = np.array([0.5, 1, 1.5], dtype=np.float64)
    groups = np.array([0, 1, 3], dtype=np.int32)
    lam = 3.0
    beta_pxgd = alo.coptimization.slope_pxgd(X, y, theta, np.zeros(3, dtype=np.float64), lam, 1e-10, 100000)
    beta_arpxgd = alo.coptimization.slope_arpxgd(X, y, theta, np.zeros(3, dtype=np.float64), lam, 1e-10, 100000)
    beta_true = np.array([-6.775 / 3.0, 0.0, 1.725], dtype=np.float64)
    assert np.abs(beta_pxgd - beta_true).max() < TOL
    assert np.abs(beta_arpxgd - beta_true).max() < TOL
    lam = 1.5
    beta_pxgd = alo.coptimization.slope_pxgd(X, y, theta, np.zeros(3, dtype=np.float64), lam, 1e-10, 100000)
    beta_arpxgd = alo.coptimization.slope_arpxgd(X, y, theta, np.zeros(3, dtype=np.float64), lam, 1e-10, 100000)
    beta_true = np.array([-9.1375 / 3.0, 0.0, 2.3625], dtype=np.float64)
    assert np.abs(beta_pxgd - beta_true).max() < TOL
    assert np.abs(beta_arpxgd - beta_true).max() < TOL
