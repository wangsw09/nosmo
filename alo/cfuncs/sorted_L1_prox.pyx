from libc.stdlib cimport malloc, free, qsort

cdef struct VAL_IDX:
    double val
    int idx

cdef int sorting_cmp(const void *a, const void *b) nogil:
    cdef VAL_IDX av = (<VAL_IDX *> a)[0]
    cdef VAL_IDX bv = (<VAL_IDX *> b)[0]
    if av.val > bv.val:
        return 1
    elif av.val < bv.val:
        return -1
    else:
        return 0

cdef void sorted_L1_prox(int p, double *x, double *theta, double tau):
    # theta is in increasing order
    cdef int k = 0 # index for loop
    cdef int t = 0 # t tracks the length of all the stack
    cdef int l = 0 # index for loop

    cdef VAL_IDX *xs = <VAL_IDX *> malloc(p * sizeof(VAL_IDX))
    cdef int *i_stk = <int *> malloc(p * sizeof(int))
    cdef int *j_stk = <int *> malloc(p * sizeof(int))
    cdef double *s_stk = <double *> malloc(p * sizeof(double))
    cdef double *w_stk = <double *> malloc(p * sizeof(double))

    for k in range(p):
        xs[k].val = abs(x[k])
        xs[k].idx = k

    qsort(xs, p, sizeof(VAL_IDX), &sorting_cmp)

    for k in range(p - 1, -1, -1):
        i_stk[t] = k
        j_stk[t] = k + 1
        s_stk[t] = abs(x[xs[k].idx]) - theta[k] * tau
        w_stk[t] = max(s_stk[t], 0.0)
        t += 1

        while (t > 1) and (w_stk[t-2] <= w_stk[t-1]):
            t -= 1
            i_stk[t-1] = i_stk[t]
            s_stk[t-1] += s_stk[t]
            w_stk[t-1] = max(s_stk[t-1] / (j_stk[t-1] - i_stk[t-1]), 0.0)

    for l in range(t):
        for k in range(i_stk[l], j_stk[l]):
            x[xs[k].idx] = w_stk[l] * ((x[xs[k].idx] > 0) * 2 - 1)
    free(xs)
    free(i_stk)
    free(j_stk)
    free(s_stk)
    free(w_stk)

