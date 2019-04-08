from libc.stdlib cimport malloc, free, qsort

cdef int sorting_cmp(const void *a, const void *b) nogil:
    cdef double av = (<double *> a)[0]
    cdef double bv = (<double *> b)[0]
    if av > bv:
        return 1
    elif av < bv:
        return -1
    else:
        return 0

cdef void L1_proj(int p, double *x, double lam):
    # proj x to w1 + ... + wp = lam
    # assume x is nonnegative, not necessarily sorted.
    cdef int i = 0
    cdef double *xx = <double *> malloc(p * sizeof(double))
    cdef double cumsum = 0.0
    cdef double theta = 0.0

    for i in range(p):
        xx[i] = abs(x[i])
        cumsum += xx[i]

    if cumsum > lam:
        qsort(xx, p, sizeof(double), &sorting_cmp)
        cumsum = -lam
        i = 0
        while (i < p) and (i * xx[i] > cumsum):
            cumsum += xx[i]
            i += 1
            theta = cumsum / i
        free(xx)
        
        for i in range(p):
            x[i] = max(abs(x[i]) - theta, 0.0) * ((x[i] > 0) * 2 - 1)

