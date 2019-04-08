from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
from scipy.sparse.linalg import svds
cimport cython

from ..clinalg.cython_blas_wrapper cimport st_dgemv, sn_dgemv, s_daxpy, mat_T_mat, s_dcopy, s_dnrm2, s_daxpy0, sn_dsyrk
from ..clinalg.cython_lapack_wrapper cimport su_dposv
from ..cprox.L1_proj cimport L1_proj

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[dtype=np.float64_t, ndim=1] _l_inf_admm(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] zumu_init,
        double lam, double rho, double abs_tol, int iter_max):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int i = 0
    cdef int iter_count = 0

    # u is dual var, y - X * beta = u
    # z = X ** T * u
    # mu is dual of (u, z); use the scaled version; mu = mu / rho
    # scaled y by rho too.
    cdef double *u = <double *> malloc(n * sizeof(double))
    cdef double *u_diff = <double *> malloc(n * sizeof(double))
    cdef double *z = <double *> malloc(p * sizeof(double))
    cdef double *z_diff = <double *> malloc(p * sizeof(double))
    cdef double *mu = <double *> malloc(p * sizeof(double))
    cdef double *mu_diff = <double *> malloc(p * sizeof(double))
    cdef np.ndarray[dtype=np.float64_t, ndim=1] ret = np.zeros(p * 2 + n, dtype=np.float64)
    
    s_dcopy(p, &zumu_init[0], z)
    s_dcopy(n, &zumu_init[p], u)
    s_dcopy(p, &zumu_init[p + n], mu)

    u_diff[0] = 10.0
    z_diff[0] = 10.0
    mu_diff[0] = 10.0

    cdef np.ndarray[dtype=np.float64_t, ndim=2] V = np.zeros((n, n), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] A = np.zeros((p + 1, n), dtype=np.float64)
    for i in range(n):
        s_daxpy0(p, rho, &X[i, 0], 1, &A[0, i], n)
    s_dcopy(n, &y[0], &A[p, 0])

    sn_dsyrk(n, p, rho, &X[0, 0], 0.0, &V[0, 0])
    for i in range(n):
        V[i, i] += 1.0
    su_dposv(n, p+1, &V[0, 0], &A[0, 0])

    while s_dnrm2(p, z_diff) > abs_tol or s_dnrm2(p, mu_diff) > abs_tol or s_dnrm2(n, u_diff) > abs_tol:
        s_dcopy(n, u, u_diff)
        s_dcopy(p, z, z_diff)
        s_dcopy(p, mu, mu_diff)

        s_dcopy(n, &A[p, 0], u)
        s_daxpy(p, -1.0, mu, z)
        st_dgemv(p, n, 1.0, &A[0, 0], z, 1.0, u)
        # u = yt + np.dot(A, z - mu)
        s_dcopy(p, mu, z)
        st_dgemv(n, p, 1.0, &X[0, 0], u, 1.0, z)
        # z = np.dot(X.T, u) + mu
        L1_proj(p, z, lam)
        s_daxpy(p, -1.0, z, mu)
        st_dgemv(n, p, 1.0, &X[0, 0], u, 1.0, mu)
        # mu += np.dot(X.T, u) - z
        s_daxpy(n, -1, u, u_diff)
        s_daxpy(p, -1, z, z_diff)
        s_daxpy(p, -1, mu, mu_diff)

        iter_count += 1
        if iter_count > iter_max:
            break

    s_dcopy(p, z, &ret[0])
    s_dcopy(n, u, &ret[p])
    s_dcopy(p, mu, &ret[p + n])

    free(u)
    free(z)
    free(mu)
    free(u_diff)
    free(z_diff)
    free(mu_diff)
    return ret

def L_inf_admm(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        double lam, double rho, double abs_tol, int iter_max):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    return _l_inf_admm(X, y, np.zeros(2 * p + n, dtype=np.float64), lam, rho, abs_tol, iter_max)

def L_inf_vec_admm(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] lams, double rho, double abs_tol, int iter_max):
    # lams must be in increasing order
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0
    
    cdef np.ndarray[dtype=np.float64_t, ndim=2] ZUMU = np.zeros((2 * p + n, k), dtype=np.float64)
    ZUMU[:, k-1] = _l_inf_admm(X, y, np.zeros(2 * p + n, dtype=np.float64), lams[k-1], rho, abs_tol, iter_max)
    for i in range(k - 2, -1, -1):
        ZUMU[:, i] = _l_inf_admm(X, y, ZUMU[:, i + 1].copy(), lams[i], rho, abs_tol, iter_max)
    return ZUMU

