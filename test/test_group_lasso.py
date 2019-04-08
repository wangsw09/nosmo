import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import alo

TOL = 1e-7
def test_group_lasso():
    abs_tol = 1e-12
    X = np.arange(15, dtype=np.float64).reshape((5, 3))
    y = np.array([0, 5, 12, -4, -8], dtype=np.float64)
    groups = np.array([0, 1, 3], dtype=np.int32)
    lam = 3.0
    beta = alo.coptimization.group_lasso_pxgd(X, y, groups, np.zeros(3, dtype=np.float64), lam, 1e-10, 100000)
    beta_true = np.array([-2.6, 0.0, 2.0], dtype=np.float64)
    assert np.abs(beta - beta_true).max() < TOL
    lam = 9.0
    beta = alo.coptimization.group_lasso_pxgd(X, y, groups, np.zeros(3, dtype=np.float64), lam, 1e-10, 100000)
    beta_true = np.array([-0.4 / 3.0, 0.0, 0.0], dtype=np.float64)
    assert np.abs(beta - beta_true).max() < TOL
