# cdef void s_dgeqrf(int m, int n, double *A_pt, double *tau)
# cdef void s_dsytrs(int n, int n_brows, double *A_pt, double *B_pt)
# for s_dlacpy, m & n is exchanged; for s_dlacpy0, m and n is not exchanged
cdef void s_dlacpy(int m, int n, double *A_pt, double *B_pt)
cdef void s_dlacpy0(int m, int n, double *A_pt, int lda, double *B_pt, int ldb)
cdef void su_dposv(int n, int nrhs, double *A_pt, double *B_pt)
cdef void sl_dposv(int n, int nrhs, double *A_pt, double *B_pt)
