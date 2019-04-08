import numpy as np
cimport numpy as np
from scipy.sparse.linalg import svds
cimport cython

from ..clinalg.cython_blas_wrapper cimport st_dgemv, sn_dgemv, s_daxpy, s_dcopy, s_dnrm2, s_ddot
from ..cprox.sorted_L1_prox cimport sorted_L1_prox

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype=np.float64_t, ndim=1] _slope_arpxgd(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] theta,
        np.ndarray[dtype=np.float64_t, ndim=1] beta_init,
        double lam, double ss, double abs_tol, int iter_max):
    # theta must be in increasing order
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int iter_count = 0
    cdef int k = 0

    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta = np.zeros(p, np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta_diff = np.ones(p, np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] betaz = np.zeros(p, np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] tmp = np.ones(n, np.float64)

    # beta_diff = beta0 - beta
    
    s_dcopy(p, &beta_init[0], &beta[0])

    while s_dnrm2(p, &beta_diff[0]) > abs_tol:
        k += 1
        s_dcopy(p, &beta[0], &betaz[0])
        s_daxpy(p,  - (k - 1) / (k + 2.0), &beta_diff[0], &betaz[0])
        s_dcopy(p, &beta[0], &beta_diff[0])
        s_dcopy(p, &betaz[0], &beta[0])

        s_dcopy(n, &y[0], &tmp[0])
        sn_dgemv(n, p, 1, &X[0, 0], &beta[0], -1, &tmp[0])
        st_dgemv(n, p, -ss, &X[0, 0], &tmp[0], 1, &beta[0])
        sorted_L1_prox(p, &beta[0], &theta[0], ss * lam )
        s_daxpy(p, -1, &beta[0], &beta_diff[0])
        s_daxpy(p, -1, &beta[0], &betaz[0])

        if s_ddot(p, &beta_diff[0], &betaz[0]) < 0:
            k = 0

        iter_count += 1
        if iter_count > iter_max:
            break
    return beta

@cython.boundscheck(False)
@cython.wraparound(False)
def slope_arpxgd(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] theta,
        np.ndarray[dtype=np.float64_t, ndim=1] beta_init,
        double lam, double abs_tol, int iter_max):
    cdef double ss = 1.0 / svds(X, k=1, which="LM", return_singular_vectors=False)[0] ** 2
    return _slope_arpxgd(X, y, theta, beta_init, lam, ss, abs_tol, iter_max)

@cython.boundscheck(False)
@cython.wraparound(False)
def slope_vec_arpxgd(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] theta,
        np.ndarray[dtype=np.float64_t, ndim=1] beta_init,
        np.ndarray[dtype=np.float64_t, ndim=1] lams, double abs_tol, int iter_max):
    # lams must be in increasing order
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef double ss = 1.0 / svds(X, k=1, which="LM", return_singular_vectors=False)[0] ** 2
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Beta = np.zeros((p, k), dtype=np.float64)
    
    Beta[:, k-1] = _slope_arpxgd(X, y, theta, beta_init, lams[k-1], ss, abs_tol, iter_max)
    for i in range(k - 2, -1, -1):
        Beta[:, i] = _slope_arpxgd(X, y, theta, Beta[:, i + 1].copy(), lams[i], ss, abs_tol, iter_max)
    return Beta

