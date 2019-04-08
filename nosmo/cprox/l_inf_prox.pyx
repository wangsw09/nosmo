from libc.stdlib cimport malloc, free, qsort
from libc.math cimport fabs

cdef struct VAL_IDX:
    double val
    int idx

cdef int sorting_cmp(const void *a, const void *b) nogil:
    cdef VAL_IDX av = (<VAL_IDX *> a)[0]
    cdef VAL_IDX bv = (<VAL_IDX *> b)[0]
    if av.val > bv.val:
        return -1
    elif av.val < bv.val:
        return 1
    else:
        return 0

cdef void L_inf_prox(int p, double *x, double tau):
    cdef int j = 0
    cdef VAL_IDX *xs = <VAL_IDX *> malloc(p * sizeof(VAL_IDX))
    for j in range(p):
        xs[j].val = fabs(x[j])
        xs[j].idx = j
    
    qsort(xs, p, sizeof(VAL_IDX), &sorting_cmp)

    cdef double xmax = xs[0].val - tau
    cdef int i = 1
    while xmax <= xs[i].val:
        xmax = (xmax * i + xs[i].val) / (i + 1.0)
        i += 1
        if i >= p:
            break
    # for j in range(i):
    #     xs[j].val = xmax

    for j in range(i):
        x[xs[j].idx] = xmax * ((x[xs[j].idx] > 0) * 2 - 1)
    free(xs)

