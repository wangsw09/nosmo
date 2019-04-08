cdef void pquad_proj(int p, double *x):
    cdef int i = 0
    for i in range(p):
        if x[i] < 0:
            x[i] = 0.0

