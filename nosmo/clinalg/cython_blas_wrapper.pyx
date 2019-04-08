from scipy.linalg.cython_blas cimport dgemv, dtrmv, daxpy, dsyrk, dcopy, dnrm2, dscal, ddot, dsyr

cdef char UTRIANG = 117
cdef char LTRIANG = 108
cdef char TRANS = 116
cdef char NOTRANS = 110
cdef int INC = 1

# s means "simplified"
# y = alpha * Ax + beta * y
cdef void st_dgemv(int m, int n,
        double alpha, double *A_pt, double *x_pt, double beta, double *y_pt):
    dgemv(&NOTRANS, &n, &m, &alpha, A_pt, &n, x_pt, &INC, &beta, y_pt, &INC)

cdef void sn_dgemv(int m, int n,
        double alpha, double *A_pt, double *x_pt, double beta, double *y_pt):
    dgemv(&TRANS, &n, &m, &alpha, A_pt, &n, x_pt, &INC, &beta, y_pt, &INC)

cdef void s_daxpy(int n, double alpha, double *x_pt, double *y_pt):
    daxpy(&n, &alpha, x_pt, &INC, y_pt, &INC)

cdef void s_daxpy0(int n, double alpha, double *x_pt, int incx, double *y_pt, int incy):
    daxpy(&n, &alpha, x_pt, &incx, y_pt, &incy)

# the upper triangle will be filled with original values of C.
cdef void st_dsyrk(int m, int n, double alpha, double *A_pt, double beta, double *C_pt):
    dsyrk(&UTRIANG, &NOTRANS, &n, &m, &alpha, A_pt, &n, &beta, C_pt, &n)

cdef void sn_dsyrk(int m, int n, double alpha, double *A_pt, double beta, double *C_pt):
    dsyrk(&UTRIANG, &TRANS, &m, &n, &alpha, A_pt, &n, &beta, C_pt, &m)

cdef void mat_T_mat(int m, int n, double *A_pt, double *C_pt):
    cdef int i = 0
    cdef int j = 0
    st_dsyrk(m, n, 1.0, A_pt, 0.0, C_pt)
    for i in range(n - 1):
        for j in range(i + 1, n):
            C_pt[i * n + j] = C_pt[j * n + i]

cdef void mat_mat_T(int m, int n, double *A_pt, double *C_pt):
    cdef int i = 0
    cdef int j = 0
    sn_dsyrk(m, n, 1.0, A_pt, 0.0, C_pt)
    for i in range(m - 1):
        for j in range(i + 1, m):
            C_pt[i * m + j] = C_pt[j * m + i]

cdef void s_dcopy(int n, double *x_pt, double *y_pt):
    dcopy(&n, x_pt, &INC, y_pt, &INC)

cdef void s_dcopy0(int n, double *x_pt, int incx, double *y_pt, int incy):
    dcopy(&n, x_pt, &incx, y_pt, &incy)

cdef double s_dnrm2(int n, double *x_pt):
    return dnrm2(&n, x_pt, &INC)

cdef void s_dscal(int n, double alpha, double *x_pt):
    dscal(&n, &alpha, x_pt, &INC)

cdef double s_ddot(int n, double *x_pt, double *y_pt):
    return ddot(&n, x_pt, &INC, y_pt, &INC)

cdef double s_ddot0(int n, double *x_pt, int incx, double *y_pt, int incy):
    return ddot(&n, x_pt, &incx, y_pt, &incy)

cdef void s_dsyr(int n, double alpha, double *x_pt, int incx, double *A_pt, int lda):
    dsyr(&UTRIANG, &n, &alpha, x_pt, &incx, A_pt, &lda)
