import numpy as np
cimport numpy as np
cimport cython

from ..clinalg.cython_blas_wrapper cimport st_dgemv, s_daxpy
from ..coptimization import L_inf_pxgd, L_inf_vec_pxgd, group_lasso_pxgd, group_lasso_vec_pxgd, slope_pxgd, slope_vec_pxgd, posv_ridge_pjgd, posv_ridge_vec_pjgd
from ..coptimization import lasso_skli, lasso_vec_skli, svm_skli, svm_vec_skli, svm_skl, svm_vec_skl, posv_ridge_pjgdi, posv_ridge_vec_pjgdi, psd_matrix_ridge_pjgd, psd_matrix_ridge_vec_pjgd

@cython.boundscheck(False)
@cython.wraparound(False)
def L_inf(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] lams,
        double abs_tol, int iter_max):
    # lams should be in increasing order.
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta_full = L_inf_pxgd(X, y, np.zeros(p, dtype=np.float64), lams[k-1], abs_tol, iter_max)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Beta_hat = np.zeros((p, k), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] loocv_pred = np.zeros((n, k), dtype=np.float64)
    mask = np.ones(n, dtype=np.bool)
    for i in range(n):
        mask[i] = False
        # here we do not need to copy X[] and y[] since boolean selection copies automatically.
        Beta_hat = L_inf_vec_pxgd(X[mask, :], y[mask], beta_full, lams, abs_tol, iter_max)
        st_dgemv(p, k, 1.0, &Beta_hat[0, 0], &X[i, 0], 0.0, &loocv_pred[i, 0])
        mask[i] = True
    return loocv_pred


@cython.boundscheck(False)
@cython.wraparound(False)
def group_lasso(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.int32_t, ndim=1] groups,
        np.ndarray[dtype=np.float64_t, ndim=1] lams,
        double abs_tol, int iter_max):
    # lams should be in increasing order.
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta_full = group_lasso_pxgd(X, y, groups,
            np.zeros(p, dtype=np.float64), lams[k-1], abs_tol, iter_max)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Beta_hat = np.zeros((p, k), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] loocv_pred = np.zeros((n, k), dtype=np.float64)
    mask = np.ones(n, dtype=np.bool)
    for i in range(n):
        mask[i] = False
        # here we do not need to copy X[] and y[] since boolean selection copies automatically.
        Beta_hat = group_lasso_vec_pxgd(X[mask, :], y[mask], groups, beta_full, lams, abs_tol, iter_max)
        st_dgemv(p, k, 1.0, &Beta_hat[0, 0], &X[i, 0], 0.0, &loocv_pred[i, 0])
        mask[i] = True
    return loocv_pred

@cython.boundscheck(False)
@cython.wraparound(False)
def slope(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] theta,
        np.ndarray[dtype=np.float64_t, ndim=1] lams,
        double abs_tol, int iter_max):
    # lams should be in increasing order.
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta_full = slope_pxgd(X, y, theta, np.zeros(p, dtype=np.float64), lams[k-1], abs_tol, iter_max)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Beta_hat = np.zeros((p, k), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] loocv_pred = np.zeros((n, k), dtype=np.float64)
    mask = np.ones(n, dtype=np.bool)
    for i in range(n):
        mask[i] = False
        # here we do not need to copy X[] and y[] since boolean selection copies automatically.
        Beta_hat = slope_vec_pxgd(X[mask, :], y[mask], theta, beta_full, lams, abs_tol, iter_max)
        st_dgemv(p, k, 1.0, &Beta_hat[0, 0], &X[i, 0], 0.0, &loocv_pred[i, 0])
        mask[i] = True
    return loocv_pred

@cython.boundscheck(False)
@cython.wraparound(False)
def posv_ridge(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] lams,
        double abs_tol, int iter_max):
    # lams should be in increasing order.
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta_full = posv_ridge_pjgd(X, y, np.zeros(p, dtype=np.float64), lams[k-1], abs_tol, iter_max)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Beta_hat = np.zeros((p, k), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] loocv_pred = np.zeros((n, k), dtype=np.float64)
    mask = np.ones(n, dtype=np.bool)
    for i in range(n):
        mask[i] = False
        # here we do not need to copy X[] and y[] since boolean selection copies automatically.
        Beta_hat = posv_ridge_vec_pjgd(X[mask, :], y[mask], beta_full, lams, abs_tol, iter_max)
        st_dgemv(p, k, 1.0, &Beta_hat[0, 0], &X[i, 0], 0.0, &loocv_pred[i, 0])
        mask[i] = True
    return loocv_pred

def lassoi( X, y, lams, abs_tol, iter_max):
    # lams should be in increasing order.
    n, p = X.shape
    k = lams.shape[0]
    loocv_pred = np.zeros((n, k), dtype=np.float64)
    mask = np.ones(n, dtype=np.bool)
    for i in range(n):
        mask[i] = False
        Beta_hat = lasso_vec_skli(X[mask, :], y[mask], lams, abs_tol, iter_max)
        loocv_pred[i, :] = np.dot(Beta_hat[1:, :].T, X[i, :]) + Beta_hat[0, :]
        mask[i] = True
    return loocv_pred

def svm( X, y, lams, abs_tol, iter_max):
    # lams should be in increasing order.
    n, p = X.shape
    k = lams.shape[0]
    loocv_pred = np.zeros((n, k), dtype=np.float64)
    mask = np.ones(n, dtype=np.bool)
    for i in range(n):
        mask[i] = False
        Beta_hat = svm_vec_skl(X[mask, :], y[mask], lams, abs_tol, iter_max)
        loocv_pred[i, :] = np.dot(Beta_hat.T, X[i, :])
        mask[i] = True
    return loocv_pred

def svmi( X, y, lams, abs_tol, iter_max):
    # lams should be in increasing order.
    n, p = X.shape
    k = lams.shape[0]
    loocv_pred = np.zeros((n, k), dtype=np.float64)
    mask = np.ones(n, dtype=np.bool)
    for i in range(n):
        mask[i] = False
        Beta_hat = svm_vec_skli(X[mask, :], y[mask], lams, abs_tol, iter_max)
        loocv_pred[i, :] = np.dot(Beta_hat[1:, :].T, X[i, :]) + Beta_hat[0, :]
        mask[i] = True
    return loocv_pred

@cython.boundscheck(False)
@cython.wraparound(False)
def posv_ridgei(
        np.ndarray[dtype=np.float64_t, ndim=2] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] lams,
        double abs_tol, int iter_max):
    # lams should be in increasing order.
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef np.ndarray[dtype=np.float64_t, ndim=1] beta_full = posv_ridge_pjgdi(X, y, np.zeros(p+1, dtype=np.float64), lams[k-1], abs_tol, iter_max)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Beta_hat = np.zeros((p+1, k), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] loocv_pred = np.zeros((n, k), dtype=np.float64)
    mask = np.ones(n, dtype=np.bool)
    for i in range(n):
        mask[i] = False
        # here we do not need to copy X[] and y[] since boolean selection copies automatically.
        Beta_hat = posv_ridge_vec_pjgdi(X[mask, :], y[mask], beta_full, lams, abs_tol, iter_max)
        st_dgemv(p, k, 1.0, &Beta_hat[1, 0], &X[i, 0], 0.0, &loocv_pred[i, 0])
        s_daxpy(k, 1.0, &Beta_hat[0, 0], &loocv_pred[i, 0])
        mask[i] = True
    return loocv_pred

@cython.boundscheck(False)
@cython.wraparound(False)
def psd_matrix_ridge(
        np.ndarray[dtype=np.float64_t, ndim=3] X,
        np.ndarray[dtype=np.float64_t, ndim=1] y,
        np.ndarray[dtype=np.float64_t, ndim=1] lams,
        double abs_tol, int iter_max):
    # lams should be in increasing order.
    cdef int n = X.shape[0]
    cdef int p = X.shape[1]
    cdef int k = lams.shape[0]
    cdef int i = 0
    cdef np.ndarray[dtype=np.float64_t, ndim=2] beta_full = psd_matrix_ridge_pjgd(X, y, np.zeros((p, p), dtype=np.float64), lams[k-1], abs_tol, iter_max)
    cdef np.ndarray[dtype=np.float64_t, ndim=3] Beta_hat = np.zeros((k, p, p), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] loocv_pred = np.zeros((n, k), dtype=np.float64)
    mask = np.ones(n, dtype=np.bool)
    for i in range(n):
        mask[i] = False
        # here we do not need to copy X[] and y[] since boolean selection copies automatically.
        Beta_hat = psd_matrix_ridge_vec_pjgd(X[mask, :, :], y[mask], beta_full, lams, abs_tol, iter_max)
        loocv_pred[i, :] = np.dot(Beta_hat.reshape(k, -1), X.reshape(n, -1)[i, :])
        mask[i] = True
    return loocv_pred

