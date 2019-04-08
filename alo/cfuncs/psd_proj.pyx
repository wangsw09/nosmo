import numpy as np
cimport numpy as np
from scipy.linalg import eigh as eigen
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def psd_proj(np.ndarray[dtype=np.float64_t, ndim=2] B):
    cdef int p = B.shape[0]
    B = (B + B.T) / 2.0
    cdef np.ndarray[dtype=np.float64_t, ndim=2] vec = np.zeros((p, p), dtype=np.float64)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] val = np.zeros(p, dtype=np.float64)
    val, vec = eigen(B, check_finite=False)
    return np.dot(vec * np.maximum(val, 0), vec.T)

