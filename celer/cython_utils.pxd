# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause
from cython cimport floating
cimport numpy as np

cdef int LASSO
cdef int LOGREG

cdef floating ST(floating, floating) nogil

cdef floating dual(int, int, floating, floating, floating *, floating *) nogil
cdef floating primal(int, floating, floating[:], floating [:],
                     floating [:], floating[:]) nogil
cdef void create_dual_pt(int, int, floating, floating *, floating *, floating *) nogil

cdef floating Nh(floating) nogil
cdef floating sigmoid(floating) nogil

cdef floating fdot(int *, floating *, int *, floating *, int *) nogil
cdef floating fasum(int *, floating *, int *) nogil
cdef void faxpy(int *, floating *, floating *, int *, floating *, int *) nogil
cdef floating fnrm2(int * , floating *, int *) nogil
cdef void fcopy(int *, floating *, int *, floating *, int *) nogil
cdef void fscal(int *, floating *, floating *, int *) nogil

cdef void fposv(char *, int *, int *, floating *,
                int *, floating *, int *, int *) nogil

cdef int create_accel_pt(
    int, int, int, int, floating, floating *, floating *,
    floating *, floating[:, :], floating[:, :], floating[:], floating[:])


cpdef void compute_Xw(
    bint, int, floating[:], floating[:],
    floating[:], bint, floating[::1, :],
    floating[:], int[:], int[:], floating[:])


cpdef void compute_norms_X_col(
    bint, floating[:], int, floating[::1, :],
    floating[:], int[:], int[:], floating[:])


cpdef floating dnorm_l1(
        bint, floating[:], floating[::1, :], floating[:],
        int[:], int[:], int[:], floating[:], floating[:], bint, bint) nogil


cdef void set_prios(
    bint, floating[:], floating[:], floating[::1, :], floating[:], int[:],
    int[:], floating[:], floating[:], floating[:], int[:], floating, int *,
    bint) nogil
