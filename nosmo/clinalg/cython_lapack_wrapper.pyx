from scipy.linalg.cython_lapack cimport dgeqrf, dlacpy, dposv

cdef char UTRIANG = 117
cdef char LTRIANG = 108
cdef char NOTRIAN = 10
cdef char TRANS = 116
cdef char NOTRANS = 110
cdef int INC = 1

# s means "simplified"

cdef void s_dgeqrf(int m, int n, double *A_pt, double *tau):
    cdef int mn = min(m, n)
    cdef double work
    cdef int lwork = -1
    cdef int info
    dgeqrf(&n, &m, A_pt, &n, tau, &work, &lwork, &info)

# A = L * D * L**T, A is symmetric, so C or F order does not matter here.
# if A is n * n, then B is m * n; after treating as Fortran order, it works.
# n_brows is number of rows of W.
cdef void su_dposv(int n, int nrhs, double *A_pt, double *B_pt): # A: pxp, B: nxp
    cdef int info
    dposv(&UTRIANG, &n, &nrhs, A_pt, &n, B_pt, &n, &info)

cdef void sl_dposv(int n, int nrhs, double *A_pt, double *B_pt): # A: pxp, B: nxp
    cdef int info
    dposv(&LTRIANG, &n, &nrhs, A_pt, &n, B_pt, &n, &info)
    
cdef void s_dlacpy(int m, int n, double *A_pt, double *B_pt):
    dlacpy(&NOTRIAN, &n, &m, A_pt, &n, B_pt, &n)

cdef void s_dlacpy0(int m, int n, double *A_pt, int lda, double *B_pt, int ldb):
    dlacpy(&NOTRIAN, &m, &n, A_pt, &lda, B_pt, &ldb)

