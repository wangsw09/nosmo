from math import sqrt
import numpy as np
cimport numpy as np
from scipy.linalg import eigh as eigen
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def psd_proj_jacob(np.ndarray[dtype=np.float64_t, ndim=2] B):
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    cdef int p = B.shape[0]
    cdef np.ndarray[dtype=np.float64_t, ndim=1] val = np.zeros(p, dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] vec = np.zeros((p, p), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] Q = np.zeros((p * p, p * (p + 1) / 2), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] A = np.zeros((p * (p + 1) / 2, p * (p + 1) / 2), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] tmp = np.zeros((p, p), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] J1 = np.zeros((p * p, p * p), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=2] J2 = np.zeros((p * p, p * p), dtype=np.float64)
    val, vec = eigen((B + B.T) / 2.0, check_finite=False)
    for i in range(p):
        Q[:, i] = np.dot(vec[:, i][:, np.newaxis], vec[:, i][np.newaxis, :]).reshape(-1)
        if val[i] > 0:
            A[i, i] = 1
    
    k = p
    for i in range(p - 1):
        for j in range(i + 1, p):
            tmp = np.dot(vec[:, i][:, np.newaxis], vec[:, j][np.newaxis, :])
            tmp = (tmp + tmp.T) / sqrt(2.0)
            Q[:, k] = tmp.reshape(-1)
            A[k, k] = (max(val[i], 0) - max(val[j], 0)) / (val[i] - val[j])
            k += 1
    J1 = np.dot(np.dot(Q, A), Q.T)
    for i in range(p):
        for j in range(p):
            if i == j:
                J2[i * p + j, i * p + j] = 1.0
            else:
                J2[i * p + j, i * p + j] = 0.5
                J2[i * p + j, j * p + i] = 0.5
    return np.dot(J1, J2)

