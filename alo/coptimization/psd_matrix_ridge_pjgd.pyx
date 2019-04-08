import numpy as np
cimport numpy as np
from scipy.sparse.linalg import svds
cimport cython

from ..clinalg.cython_blas_wrapper cimport st_dgemv, sn_dgemv, s_daxpy, mat_T_mat, s_dcopy, s_dnrm2
from ..cfuncs import psd_proj

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype=np.float64_t, ndim=1] _psd_matrix_ridge_pjgd(
        np.ndarray[dtype=np.float64_t, ndim=3] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=2] beta_init,
        double lam, double ss, double abs_tol, int iter_max):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int iter_count = 0

    cdef np.ndarray[dtype=np.float64_t, ndim=2] beta = np.zeros((p, p), np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] beta0 = np.ones((p, p), np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] tmp = np.ones(n, np.float64)
    
    s_dcopy(p * p, &beta_init[0, 0], &beta[0, 0])

    while s_dnrm2(p * p, &beta0[0, 0]) > abs_tol:
        s_dcopy(p * p, &beta[0, 0], &beta0[0, 0])
        s_dcopy(n, &y[0], &tmp[0])
        sn_dgemv(n, p * p, 1, &X[0, 0, 0], &beta0[0, 0], -1, &tmp[0])
        st_dgemv(n, p * p, -ss, &X[0, 0, 0], &tmp[0], 1.0 - ss * 2.0 * lam, &beta[0, 0])
        beta = psd_proj(beta)
        s_daxpy(p * p, -1, &beta[0, 0], &beta0[0, 0])

        iter_count += 1
        if iter_count > iter_max:
            break
    return beta

@cython.boundscheck(False)
@cython.wraparound(False)
def psd_matrix_ridge_pjgd(
        np.ndarray[dtype=np.float64_t, ndim=3] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=2] beta_init,
        double lam, double abs_tol, int iter_max):
    cdef int n = X.shape[0]
    cdef double ss = 1.0 / (svds(X.reshape((n, -1)), k=1, which="LM", return_singular_vectors=False)[0] ** 2 + 2 * lam)
    return _psd_matrix_ridge_pjgd(X, y, beta_init, lam, ss, abs_tol, iter_max)

@cython.boundscheck(False)
@cython.wraparound(False)
def psd_matrix_ridge_vec_pjgd(
        np.ndarray[dtype=np.float64_t, ndim=3] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=2] beta_init,
        np.ndarray[dtype=np.float64_t, ndim=1] lams, double abs_tol, int iter_max):
    # lams must be in increasing order
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef double inv_ss = svds(X.reshape((n, -1)), k=1, which="LM", return_singular_vectors=False)[0] ** 2
    cdef np.ndarray[dtype=np.float64_t, ndim=3] Beta = np.zeros((k, p, p), dtype=np.float64)
    
    Beta[k-1, :, :] = _psd_matrix_ridge_pjgd(X, y, beta_init, lams[k-1], 1.0 / (inv_ss + 2 * lams[k-1]), abs_tol, iter_max)
    for i in range(k - 2, -1, -1):
        Beta[i, :, :] = _psd_matrix_ridge_pjgd(X, y, Beta[i+1, :, :].copy(), lams[i], 1.0 / (inv_ss + 2 * lams[i]), abs_tol, iter_max)
    return Beta

