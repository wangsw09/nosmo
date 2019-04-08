from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport cython

from ..clinalg.cython_blas_wrapper cimport st_dgemv, sn_dgemv, s_daxpy, mat_T_mat, s_dcopy, s_dcopy0, s_dnrm2, s_ddot, s_dscal, s_daxpy0, st_dsyrk
from ..clinalg.cython_lapack_wrapper cimport s_dlacpy, su_dposv
from ..cfuncs.pquad_proj cimport pquad_proj
from ..cfuncs import psd_proj
from ..cfuncs import psd_proj_jacob

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _alo_psd_matrix_ridge(
        np.ndarray[dtype=np.float64_t, ndim=3] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=2] beta_hat,
        double lam, double thresh_tol,
        np.ndarray[dtype=np.float64_t, ndim=1] tmp_n):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int i = 0
    cdef int j = 0
    
    cdef np.ndarray[dtype=np.float64_t, ndim=2] beta_hat0 = (1.0 - 2.0 * lam) * beta_hat - np.sum(X * (np.sum(X * beta_hat, axis=(1, 2)) - y)[:, np.newaxis, np.newaxis], axis=0)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] J = psd_proj_jacob(beta_hat0)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] G = np.zeros((p*p, p*p), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] GX = np.zeros((p*p, n), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] beta_tmp = np.zeros((p, p), dtype=np.float64)

    G = np.linalg.solve(np.dot(J, np.dot(X.reshape(n, -1).T, X.reshape(n, -1))) + np.eye(p*p) + (2.0 * lam - 1.0) * J, J)
    GX = np.dot(G, X.reshape(n, -1).T)
    for i in range(n):
        beta_tmp = beta_hat + GX[:, i].reshape((p, p)) * (np.sum(X[i, :] * beta_hat) - y[i]) / (1.0 - np.dot(X.reshape((n, -1))[i, :], GX[:, i]))
        beta_tmp = psd_proj(beta_tmp)
        tmp_n[i] = np.sum(X[i, :] * beta_tmp)

@cython.boundscheck(False)
@cython.wraparound(False)
def alo_psd_matrix_ridge( np.ndarray[dtype=np.float64_t, ndim=3] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=3] Beta_hat,
        np.ndarray[dtype=np.float64_t, ndim=1] lams,
        double thresh_tol):
    cdef int n = X.shape[0]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Alo_pred = np.zeros((n, k), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] tmp_n = np.zeros(n, dtype=np.float64)
    for i in range(k):
        # must use Beta_hat[:, i].copy(); otherwise passed array is not contiguous!
        _alo_psd_matrix_ridge(X, y, Beta_hat[i, :, :].copy(), lams[i], thresh_tol, tmp_n)
        s_dcopy0(n, &tmp_n[0], 1, &Alo_pred[0, i], k)
    return Alo_pred

