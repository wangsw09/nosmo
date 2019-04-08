import numpy as np
cimport numpy as np
cimport cython

from ..clinalg.cython_blas_wrapper cimport st_dgemv, sn_dgemv, s_daxpy, mat_T_mat, s_dcopy, s_dcopy0, s_dnrm2, s_ddot, s_dscal, s_daxpy0, st_dsyrk, s_dsyr
from ..clinalg.cython_lapack_wrapper cimport s_dlacpy, s_dlacpy0, su_dposv

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _alo_group_lasso(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.int32_t, ndim=1] groups,
        np.ndarray[dtype=np.float64_t, ndim=1] beta_hat,
        double lam, double thresh_tol,
        np.ndarray[dtype=np.float64_t, ndim=1] tmp_n):
    # groups = [0, ..., p], of which the component is the beginning of the next block.
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = groups.shape[0] - 1
    cdef int i = 0
    cdef int j = 0
    cdef int l = 0
    cdef int num_nz_grp = 0
    
    cdef np.ndarray[dtype=np.float64_t, ndim=1] grp_norm = np.zeros(k, dtype=np.float64) # container len=p
    for i in range(k):
        grp_norm[i] = s_dnrm2(groups[i + 1] - groups[i], &beta_hat[groups[i]])
        if grp_norm[i] > thresh_tol:
            num_nz_grp += groups[i + 1] - groups[i]

    cdef np.ndarray[dtype=np.float64_t, ndim=2] W = np.zeros((n, num_nz_grp), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] W_cpy = np.zeros((n, num_nz_grp), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] WTW = np.zeros((num_nz_grp, num_nz_grp), dtype=np.float64)

    for i in range(k):
        if grp_norm[i] > thresh_tol:
            s_dlacpy0(groups[i+1] - groups[i], n, &X[0, groups[i]], p, &W[0, j], num_nz_grp)
            j += groups[i+1] - groups[i]
    s_dlacpy(n, num_nz_grp, &W[0, 0], &W_cpy[0, 0])

    # calculating H matrix
    st_dsyrk(n, num_nz_grp, 1.0, &W[0, 0], 0.0, &WTW[0, 0])

    j = 0
    for i in range(k):
        if grp_norm[i] > thresh_tol:
            s_dsyr(groups[i+1] - groups[i], - lam / grp_norm[i] ** 3, &beta_hat[groups[i]], 1, &WTW[j, j], num_nz_grp)
            j += groups[i+1] - groups[i]

    j = 0
    for i in range(k):
        if grp_norm[i] > thresh_tol:
            for l in range(j, j + groups[i + 1] - groups[i]):
                WTW[l, l] += lam / grp_norm[i]
            j += groups[i+1] - groups[i]

    su_dposv(num_nz_grp, n, &WTW[0, 0], &W[0, 0])

    # calculate diagonal of H and the ALO leave-i-prediction
    cdef double tmp = 0.0
    for i in range(n):
        tmp_n[i] = s_ddot(num_nz_grp, &W_cpy[i, 0], &W[i, 0]) # diagonal element H_ii
        tmp = s_ddot(p, &X[i, 0], &beta_hat[0])
        tmp_n[i] = tmp + tmp_n[i] / (1.0 - tmp_n[i]) * (tmp - y[i])

@cython.boundscheck(False)
@cython.wraparound(False)
def alo_group_lasso( np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.int32_t, ndim=1] groups,
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
        _alo_group_lasso(X, y, groups, Beta_hat[:, i].copy(), lams[i], thresh_tol, tmp_n)
        s_dcopy0(n, &tmp_n[0], 1, &Alo_pred[0, i], k)
    return Alo_pred

