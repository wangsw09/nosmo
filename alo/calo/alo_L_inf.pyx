from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport cython

from ..clinalg.cython_blas_wrapper cimport st_dgemv, sn_dgemv, s_daxpy, mat_T_mat, s_dcopy, s_dcopy0, s_dnrm2, s_ddot, s_dscal, s_daxpy0, st_dsyrk
from ..clinalg.cython_lapack_wrapper cimport s_dlacpy, su_dposv

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _alo_L_inf(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] beta_hat,
        double lam, double thresh_tol,
        np.ndarray[dtype=np.float64_t, ndim=1] tmp_n):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int i = 0
    cdef int j = 0
    
    # x^T * beta
    cdef np.ndarray[dtype=np.float64_t, ndim=1] tmp_p = np.zeros(p, dtype=np.float64) # container len=p
    
    s_dcopy(n, &y[0], &tmp_n[0])
    sn_dgemv(n, p, -1.0, &X[0, 0], &beta_hat[0], 1.0, &tmp_n[0])
    st_dgemv(n, p, 1.0, &X[0, 0], &tmp_n[0], 0.0, &tmp_p[0])

    # tmp_p = X^T * (y - X * beta_hat)

    cdef int w_dim = 0
    for i in range(p):
        if abs(tmp_p[i]) <= thresh_tol:
            w_dim += 1

    print (w_dim + 1)

    s_dscal(n, 0.0, &tmp_n[0])
    cdef np.ndarray[dtype=np.float64_t, ndim=2] W = np.zeros((n, w_dim + 1), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] W_cpy = np.zeros((n, w_dim + 1), dtype=np.float64)

    for i in range(p):
        if abs(tmp_p[i]) <= thresh_tol:
            s_dcopy0(n, &X[0, i], p, &W[0, j], w_dim + 1)
            j += 1
        else:
            s_daxpy0(n, (tmp_p[i] > 0) * 2.0 - 1.0, &X[0, i], p, &tmp_n[0], 1)
    s_dcopy0(n, &tmp_n[0], 1, &W[0, j], w_dim + 1)
    s_dlacpy(n, w_dim+1, &W[0, 0], &W_cpy[0, 0])

    # calculating H matrix
    cdef np.ndarray[dtype=np.float64_t, ndim=2] WTW = np.zeros((w_dim + 1, w_dim + 1), dtype=np.float64)
    st_dsyrk(n, w_dim + 1, 1.0, &W[0, 0], 0.0, &WTW[0, 0])
    su_dposv(w_dim + 1, n, &WTW[0, 0], &W[0, 0])

    # calculate diagonal of H and the ALO leave-i-prediction
    cdef double tmp = 0.0
    for i in range(n):
        tmp_n[i] = s_ddot(w_dim + 1, &W_cpy[i, 0], &W[i, 0]) # diagonal element H_ii
        tmp = s_ddot(p, &X[i, 0], &beta_hat[0])
        tmp_n[i] = tmp + tmp_n[i] / (1.0 - tmp_n[i]) * (tmp - y[i])

@cython.boundscheck(False)
@cython.wraparound(False)
def alo_L_inf( np.ndarray[dtype=np.float64_t, ndim=2] X,
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
        _alo_L_inf(X, y, Beta_hat[:, i].copy(), lams[i], thresh_tol, tmp_n)
        s_dcopy0(n, &tmp_n[0], 1, &Alo_pred[0, i], k)
    return Alo_pred


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _alo_L_inf_dual(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] zu,
        double lam, double thresh_tol,
        np.ndarray[dtype=np.float64_t, ndim=1] tmp_n):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int i = 0
    cdef int j = 0
    
    # tmp_p = X^T * (y - X * beta_hat)
    cdef int w_dim = 0
    for i in range(p):
        if abs(zu[i]) <= thresh_tol:
            w_dim += 1

    print (w_dim + 1)

    s_dscal(n, 0.0, &tmp_n[0])
    cdef np.ndarray[dtype=np.float64_t, ndim=2] W = np.zeros((n, w_dim + 1), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] W_cpy = np.zeros((n, w_dim + 1), dtype=np.float64)

    for i in range(p):
        if abs(zu[i]) <= thresh_tol:
            s_dcopy0(n, &X[0, i], p, &W[0, j], w_dim + 1)
            j += 1
        else:
            s_daxpy0(n, (zu[i] > 0) * 2.0 - 1.0, &X[0, i], p, &tmp_n[0], 1)
    s_dcopy0(n, &tmp_n[0], 1, &W[0, j], w_dim + 1)
    s_dlacpy(n, w_dim+1, &W[0, 0], &W_cpy[0, 0])

    # calculating H matrix
    cdef np.ndarray[dtype=np.float64_t, ndim=2] WTW = np.zeros((w_dim + 1, w_dim + 1), dtype=np.float64)
    st_dsyrk(n, w_dim + 1, 1.0, &W[0, 0], 0.0, &WTW[0, 0])
    su_dposv(w_dim + 1, n, &WTW[0, 0], &W[0, 0])

    # calculate diagonal of H and the ALO leave-i-prediction
    cdef double tmp = 0.0
    for i in range(n):
        tmp_n[i] = s_ddot(w_dim + 1, &W_cpy[i, 0], &W[i, 0]) # diagonal element H_ii
        tmp = zu[i + p]
        tmp_n[i] = y[i] - tmp - tmp_n[i] / (1.0 - tmp_n[i]) * tmp

@cython.boundscheck(False)
@cython.wraparound(False)
def alo_L_inf_dual( np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=2] ZUMU,
        np.ndarray[dtype=np.float64_t, ndim=1] lams,
        double thresh_tol):
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Alo_pred = np.zeros((n, k), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] tmp_n = np.zeros(n, dtype=np.float64)
    for i in range(k):
        # must use Beta_hat[:, i].copy(); otherwise passed array is not contiguous!
        _alo_L_inf_dual(X, y, ZUMU[: (n + p), i].copy(), lams[i], thresh_tol, tmp_n)
        s_dcopy0(n, &tmp_n[0], 1, &Alo_pred[0, i], k)
    return Alo_pred

