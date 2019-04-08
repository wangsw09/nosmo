from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
from scipy.sparse.linalg import svds
cimport cython

from ..clinalg.cython_blas_wrapper cimport st_dgemv, sn_dgemv, s_daxpy, s_dcopy, s_dnrm2, s_dscal
from ..cfuncs.sorted_L1_prox cimport sorted_L1_prox

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype=np.float64_t, ndim=1] _slope_pxgd(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        double *y,
        double *theta,
        double *beta_init,
        np.ndarray[dtype=np.float64_t, ndim=1] beta,
        double lam, double ss, double abs_tol, int iter_max):
    # theta must be in increasing order
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int iter_count = 0

    cdef double *beta0 = <double *> malloc(p * sizeof(double))
    cdef double *tmp = <double *> malloc(n * sizeof(double))
    
    s_dscal(p, 0.0, beta0)
    beta0[0] += 1.0

    s_dcopy(p, beta_init, &beta[0])

    while s_dnrm2(p, beta0) > abs_tol:
        s_dcopy(p, &beta[0], beta0)
        s_dcopy(n, y, tmp)
        sn_dgemv(n, p, 1, &X[0, 0], beta0, -1, tmp)
        st_dgemv(n, p, -ss, &X[0, 0], tmp, 1, &beta[0])
        sorted_L1_prox(p, &beta[0], theta, ss * lam )
        s_daxpy(p, -1, &beta[0], beta0)

        iter_count += 1
        if iter_count > iter_max:
            break
    free(beta0)
    free(tmp)
    return beta

@cython.boundscheck(False)
@cython.wraparound(False)
def slope_pxgd(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] theta,
        np.ndarray[dtype=np.float64_t, ndim=1] beta_init,
        double lam, double abs_tol, int iter_max):
    cdef double ss = 1.0 / svds(X, k=1, which="LM", return_singular_vectors=False)[0] ** 2
    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta = np.zeros(X.shape[1], np.float64)
    _slope_pxgd(X, &y[0], &theta[0], &beta_init[0], beta, lam, ss, abs_tol, iter_max)
    return beta

@cython.boundscheck(False)
@cython.wraparound(False)
def slope_vec_pxgd(
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
    cdef np.ndarray[dtype=np.float64_t, ndim=2, mode="fortran"] Beta = np.zeros((p, k), dtype=np.float64, order="F")
    
    _slope_pxgd(X, &y[0], &theta[0], &beta_init[0], Beta[:, k-1], lams[k-1], ss, abs_tol, iter_max)
    for i in range(k - 2, -1, -1):
        _slope_pxgd(X, &y[0], &theta[0], &Beta[0, i + 1], Beta[:, i], lams[i], ss, abs_tol, iter_max)
    return np.ascontiguousarray(Beta)

