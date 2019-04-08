from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport cython

from ..clinalg.cython_blas_wrapper cimport st_dgemv, sn_dgemv, s_daxpy, mat_T_mat, s_dcopy, s_dcopy0, s_dnrm2, s_ddot, s_dscal, s_daxpy0, st_dsyrk
from ..clinalg.cython_lapack_wrapper cimport s_dlacpy, su_dposv
from ..cfuncs.pquad_proj cimport pquad_proj

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _alo_posv_ridge(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] beta_hat,
        double lam, double thresh_tol,
        np.ndarray[dtype=np.float64_t, ndim=1] tmp_n):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int i = 0
    cdef int j = 0
    
    cdef int w_dim = 0
    for i in range(p):
        if beta_hat[i] >= thresh_tol:
            w_dim += 1

    cdef np.ndarray[dtype=np.float64_t, ndim=2] Gamma = np.zeros((p, w_dim), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] XGamma = np.zeros((n, w_dim), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] GX = np.zeros((p, n), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta_tmp = np.zeros(p, dtype=np.float64)
    j = 0
    for i in range(p):
        if beta_hat[i] >= thresh_tol:
            Gamma[i, j] = 1.0
            XGamma[:, j] = X[:, i]
            j += 1
    GX = np.dot(Gamma, np.linalg.solve(np.dot(XGamma.T, XGamma) + 2.0 * lam * np.eye(w_dim), XGamma.T))
    for i in range(n):
        beta_tmp = beta_hat + GX[:, i] * (np.dot(X[i, :], beta_hat) - y[i]) / (1.0 - np.dot(X[i, :], GX[:, i]))
        pquad_proj(p, &beta_tmp[0])
        tmp_n[i] = np.dot(X[i, :], beta_tmp)        

@cython.boundscheck(False)
@cython.wraparound(False)
def alo_posv_ridge( np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=2] Beta_hat,
        np.ndarray[dtype=np.float64_t, ndim=1] lams,
        double thresh_tol):
    cdef int n = X.shape[0]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Alo_pred = np.zeros((n, k), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] tmp_n = np.zeros(n, dtype=np.float64)
    for i in range(k):
        # must use Beta_hat[:, i].copy(); otherwise passed array is not contiguous!
        _alo_posv_ridge(X, y, Beta_hat[:, i].copy(), lams[i], thresh_tol, tmp_n)
        s_dcopy0(n, &tmp_n[0], 1, &Alo_pred[0, i], k)
    return Alo_pred

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _alo_posv_ridgei(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] beta_hat,
        double lam, double thresh_tol,
        np.ndarray[dtype=np.float64_t, ndim=1] tmp_n):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int i = 0
    cdef int j = 0
    
    cdef int w_dim = 0
    for i in range(p):
        if beta_hat[i+1] >= thresh_tol:
            w_dim += 1

    cdef np.ndarray[dtype=np.float64_t, ndim=2] Gamma = np.zeros((p, w_dim), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] XGamma = np.zeros((n, w_dim), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] GX = np.zeros((p, n), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta_tmp = np.zeros(p + 1, dtype=np.float64)
    j = 0
    for i in range(p):
        if beta_hat[i+1] >= thresh_tol:
            Gamma[i, j] = 1.0
            XGamma[:, j] = X[:, i]
            j += 1
    GX = np.dot(Gamma, np.linalg.solve(np.dot(XGamma.T, XGamma) + 2.0 * lam * np.eye(w_dim), XGamma.T))
    XGX = np.dot(X, GX)
    abGb = n - np.sum(XGX)
    d = 1.0 - np.dot(XGX, np.ones(n, dtype=np.float64))
    XGX0 = XGX + np.dot(d[:, np.newaxis], d[np.newaxis, :]) / abGb
    GX0 = GX + (-np.dot(np.dot(GX, np.ones(n))[:, np.newaxis], np.ones(n)[np.newaxis, :]) + np.dot(np.dot(GX, np.ones(n))[:, np.newaxis], np.dot(XGX, np.ones(n))[np.newaxis, :])) / abGb
    GX00 = (1.0 - np.dot(XGX, np.ones(n))) / abGb
    for i in range(n):
        beta_tmp[1:] = beta_hat[1:] + GX0[:, i] * (np.dot(X[i, :], beta_hat[1:]) + beta_hat[0] - y[i]) / (1.0 - XGX0[i, i])
        beta_tmp[0] = beta_hat[0] + GX00[i] * (np.dot(X[i, :], beta_hat[1:]) + beta_hat[0] - y[i]) / (1.0 - XGX0[i, i])
        pquad_proj(p, &beta_tmp[1])
        tmp_n[i] = np.dot(X[i, :], beta_tmp[1:]) + beta_tmp[0]      

@cython.boundscheck(False)
@cython.wraparound(False)
def alo_posv_ridgei( np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=2] Beta_hat,
        np.ndarray[dtype=np.float64_t, ndim=1] lams,
        double thresh_tol):
    cdef int n = X.shape[0]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Alo_pred = np.zeros((n, k), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] tmp_n = np.zeros(n, dtype=np.float64)
    for i in range(k):
        # must use Beta_hat[:, i].copy(); otherwise passed array is not contiguous!
        _alo_posv_ridgei(X, y, Beta_hat[:, i].copy(), lams[i], thresh_tol, tmp_n)
        s_dcopy0(n, &tmp_n[0], 1, &Alo_pred[0, i], k)
    return Alo_pred

