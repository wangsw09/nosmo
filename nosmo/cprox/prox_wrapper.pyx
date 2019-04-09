import numpy as np
cimport numpy as np
cimport cython

from .sorted_L1_prox cimport sorted_L1_prox

@cython.boundscheck(False)
@cython.wraparound(False)
def slope_prox(np.ndarray[dtype=np.float64_t, ndim=1] x, np.ndarray[dtype=np.float64_t, ndim=1] theta, double tau):
    cdef int p = x.shape[0]
    sorted_L1_prox(p, &x[0], &theta[0], tau)

