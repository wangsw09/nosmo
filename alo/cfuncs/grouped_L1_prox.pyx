from ..clinalg.cython_blas_wrapper cimport s_dnrm2, s_dscal

cdef void grouped_L1_prox(double *x, int k, int *groups, double tau):
    # k is the number of groups, which equals the length of array groups - 1;
    # groups = [0, ..., p], of which the component is the beginning of the next block.
    cdef int i = 1
    cdef double loc_norm = 0
    
    for i in range(k):
        loc_norm = s_dnrm2(groups[i + 1] - groups[i], x + groups[i])
        if loc_norm == 0:
            s_dscal(groups[i + 1] - groups[i], 0, x + groups[i])
        else:
            s_dscal(groups[i + 1] - groups[i], max(1.0 - tau / loc_norm, 0.0), x + groups[i])

