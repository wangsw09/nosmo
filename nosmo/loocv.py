import numpy as np
import numpy.linalg as npla

from .loss_reg import *
from . import gd
import optim_algo as oa

def loo_glm_ridge(y, X, lam_seq=None, intercept=False,
        family="gaussian", err_func=lambda y1, y2: (y1 - y2) ** 2 ):
    n, p = X.shape
    m = lam_seq.shape[0]
    loo_err = np.zeros(m)

    mask = np.ones(n, dtype=np.bool)

    if intercept is False:
        for i in xrange(n):
            mask[i] = False
            beta_mat = oa.glm_ridge(y[mask], X[mask, :], lam_seq=lam_seq, intercept=False,
                family=family)
            pred = np.dot(X[i, :], beta_mat)
            loo_err += err_func(y[i], pred)
            mask[i] = True
        return loo_err / n
    else:
        raise NotImplementedError

def loo_glm_lasso(y, X, lam_seq=None, intercept=False,
        family="gaussian", err_func=lambda y1, y2: (y1 - y2) ** 2, tol=1e-4,
        max_iter=50000):
    n, p = X.shape
    m = lam_seq.shape[0]
    loo_err = np.zeros(m)

    mask = np.ones(n, dtype=np.bool)

    if intercept is False:
        for i in xrange(n):
            mask[i] = False
            beta_mat = oa.glm_lasso(y[mask], X[mask, :], lam_seq=lam_seq, intercept=False,
                family=family, tol=tol, max_iter=max_iter)
            pred = np.dot(X[i, :], beta_mat)
            loo_err += err_func(y[i], pred)
            mask[i] = True
        return loo_err / n

    else:
        raise NotImplementedError

def loo_L_inf(y, X, lam_seq=None, intercept=False,
        family="gaussian", err_func=lambda y1, y2: (y1 - y2) ** 2,
        max_iter=1000, tol=1e-7):
    n, p = X.shape
    m = lam_seq.shape[0]
    loo_err = np.zeros(m)

    mask = np.ones(n, dtype=np.bool)

    if intercept is False:
        for j in xrange(m):
            for i in xrange(n):
                mask[i] = False
                beta = gd.model.linear_L_inf(y[mask], X[mask, :],
                        lam=lam_seq[j],
                    max_iter=max_iter, tol=tol, algo="acc_prox_gd_restart")
                pred = np.dot(X[i, :], beta)
                loo_err[j] += err_func(y[i], pred)
                mask[i] = True
        return loo_err / n

    else:
        raise NotImplementedError

def loo_svm(y, X, lam_seq=None, intercept=False, err_func=lambda y1, y2: (y1 - y2) ** 2 ):
    n, p = X.shape
    m = lam_seq.shape[0]
    loo_err = np.zeros(m)

    mask = np.ones(n, dtype=np.bool)
    if intercept is False:
        for j in xrange(m):
            for i in xrange(n):
                mask[i] = False
                beta = oa.svm_linear(y[mask], X[mask, :], lam=lam_seq[j], intercept=False)
                pred = np.dot(X[i, :], beta)
                loo_err[j] += err_func(y[i], pred)
                mask[i] = True
        return loo_err / n

    else:
        raise NotImplementedError

def loo_pst_ridge(y, X, lam_seq=None, intercept=False, err_func=lambda y1, y2:
        (y1 - y2) ** 2, max_iter=2000, tol=1e-7 ):
    n, p = X.shape
    m = lam_seq.shape[0]
    loo_err = np.zeros(m)

    mask = np.ones(n, dtype=np.bool)
    if intercept is False:
        for j in xrange(m):
            for i in xrange(n):
                mask[i] = False
                beta = gd.model.linear_pst_ridge(y[mask], X[mask, :],
                        lam=lam_seq[j], max_iter=max_iter,
                tol=tol, algo="acc_proj_gd_restart")

                pred = np.dot(X[i, :], beta)
                loo_err[j] += err_func(y[i], pred)
                mask[i] = True
        return loo_err / n

    else:
        raise NotImplementedError

