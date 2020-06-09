#cython: language_level=3
# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import numpy as np
import time

from numpy.linalg import norm

cimport numpy as np
cimport cython

from cython cimport floating
from libc.math cimport fabs, sqrt, exp

from .cython_utils cimport fdot, faxpy, fcopy, fposv, fscal, fnrm2
from .cython_utils cimport (primal, dual, create_dual_pt, create_accel_pt,
                           sigmoid, ST, LOGREG)

cdef:
    int inc = 1

# TODO move this to utils.pyx
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating dual_norm(
    bint is_sparse, int n_features, int n_samples,
    floating * theta, floating[::1, :] X,
    floating[:] X_data, int[:] X_indices,
    int[:] X_indptr, int ws_size, long * C) nogil:
    """Compute norm(X.T.dot(theta), ord=inf),
    with X restricted to features (columns) with indices in array C.
    if ws_size == n_features, C=np.arange(n_features is used)"""
    cdef floating Xj_theta
    cdef floating scal = 0.
    cdef int j
    cdef int Cj
    cdef int i
    cdef int startptr, endptr
    if ws_size == n_features:  # scaling wrt all features
        for j in range(n_features):
            if is_sparse:
                startptr = X_indptr[j]
                endptr = X_indptr[j + 1]
                Xj_theta = 0.
                for i in range(startptr, endptr):
                    Xj_theta += X_data[i] * theta[X_indices[i]]
            else:
                Xj_theta = fdot(&n_samples, &X[0, j], &inc,
                                &theta[0], &inc)

            scal = max(scal, fabs(Xj_theta))
    else:  # scaling wrt features in C only
        for j in range(ws_size):
            Cj = C[j]
            if is_sparse:
                startptr = X_indptr[Cj]
                endptr = X_indptr[Cj + 1]
                Xj_theta = 0.
                for i in range(startptr, endptr):
                    Xj_theta += X_data[i] * theta[X_indices[i]]
            else:
                Xj_theta = fdot(&n_samples, &X[0, Cj], &inc,
                                &theta[0], &inc)
            scal = max(scal, fabs(Xj_theta))

    return scal


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void compute_norms(
        floating[:] norms_X_col, bint is_sparse, int n_features, int n_samples,
        floating[::1, :] X, floating[:] X_data, int[:] X_indices,
        int[:] X_indptr) nogil:
    """Compute norms_X_col in place."""
    cdef int j, startptr, endptr
    cdef double tmp
    if is_sparse:
        for j in range(n_features):
            tmp = 0
            startptr = X_indptr[j]
            endptr = X_indptr[j + 1]
            for i in range(startptr, endptr):
                tmp += X_data[i] ** 2
            norms_X_col[j] = sqrt(tmp)
    else:
        for j in range(n_features):
            norms_X_col[j] = fnrm2(&n_samples, &X[0, j], &inc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void set_feature_prios(
        floating[:] prios, bint is_sparse, int n_features, int n_samples,
        floating[::1, :] X, floating[:] X_data, int[:] X_indices,
        int[:] X_indptr, floating[:] theta, floating[:] norms_X_col):
    """Fill the array prios with priorities per feature."""
    cdef int j, startptr, endptr
    cdef double Xj_theta
    for j in range(n_features):
        if is_sparse:
            Xj_theta = 0.
            startptr = X_indptr[j]
            endptr = X_indptr[j + 1]
            for i in range(startptr, endptr):
                Xj_theta += theta[X_indices[i]] * X_data[i]
        else:
            Xj_theta = fdot(&n_samples, &X[0, j], &inc, &theta[0], &inc)

        prios[j] = (1. - fabs(Xj_theta)) / norms_X_col[j]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void compute_Xw(
        floating[:] Xw, floating[:] w, bint is_sparse, floating[::1, :] X,
        floating[:] X_data, int[:] X_indices, int[:] X_indptr) nogil:
    """Compute X @ w in place."""
    # TODO this function is available in cython_utils with sparse scaling
    cdef int j
    cdef int startptr, endptr
    cdef int n_features = w.shape[0]
    cdef int n_samples = Xw.shape[0]

    for j in range(n_features):
        if w[j] != 0:
            if is_sparse:
                startptr = X_indptr[j]
                endptr = X_indptr[j + 1]
                for i in range(startptr, endptr):
                    Xw[X_indices[i]] += w[j] * X_data[i]
            else:
                faxpy(&n_samples, &w[j], &Xw[0], &inc, &X[0, j], &inc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def newton_celer(
        bint is_sparse, floating[::1, :] X, floating[:] X_data,
        int[:] X_indices, int[:] X_indptr, floating[:] y, floating alpha,
        floating[:] w, int max_iter, floating tol=1e-4, int p0=100,
        int verbose=0, bint use_accel=1, bint prune=1, bint blitz_sc=False):

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int verbose_in = max(0, verbose - 1)

    cdef int i, j, t, k
    cdef floating p_obj, d_obj, gap, norm_Xtheta, norm_Xtheta_acc
    cdef floating tmp
    cdef int info_dposv
    cdef int ws_size
    cdef floating eps_inner = 0.1
    cdef floating growth = 2.

    cdef int n_samples = y.shape[0]
    cdef int n_features = w.shape[0]
    # cdef int[:] all_features = np.arange(n_features, dtype=np.int32)
    cdef long[:] all_features = np.arange(n_features)
    cdef floating[:] prios = np.empty(n_features, dtype=dtype)
    cdef long[:] WS  # hacky for now TODO fix, int causing runtime error
    cdef floating[:] gaps = np.zeros(max_iter, dtype=dtype)

    cdef floating d_obj_acc = 0.
    cdef floating tol_inner


    cdef int K = 6
    cdef floating[:, :] last_K_Xw = np.zeros((K, n_samples), dtype=dtype)
    cdef floating[:, :] U = np.zeros((K - 1, n_samples), dtype=dtype)
    cdef floating[:, :] UUt = np.zeros((K - 1, K - 1), dtype=dtype)
    cdef floating[:] onesK = np.ones((K - 1), dtype=dtype)

    cdef floating[:] norms_X_col = np.zeros(n_features, dtype=dtype)
    compute_norms(norms_X_col, is_sparse, n_features, n_samples, X,
                  X_data, X_indices, X_indptr)
    cdef floating[:] Xw = np.zeros(n_samples, dtype=dtype)
    compute_Xw(Xw, w, is_sparse, X, X_data, X_indices, X_indptr)

    cdef floating[:] theta = np.empty(n_samples, dtype=dtype)
    cdef floating[:] theta_acc = np.empty(n_samples, dtype=dtype)

    cdef floating[:] exp_Xw = np.empty(n_samples, dtype=dtype)
    for i in range(n_samples):
        exp_Xw[i] = exp(Xw[i])
    cdef floating[:] low_exp_Xw = np.empty(n_samples, dtype=dtype)
    cdef floating[:] aux = np.empty(n_samples, dtype=dtype)
    cdef int[:] is_positive_label = np.empty(n_samples, dtype=np.int32)
    for i in range(n_samples):
        if y[i] > 0.:
            is_positive_label[i] = 1
        else:
            is_positive_label[i] = 0

    cdef char * char_U = 'U'
    cdef int one = 1
    cdef int Kminus1 = K - 1
    cdef floating sum_z

    for t in range(max_iter):
        p_obj = primal(LOGREG, alpha, n_samples, &Xw[0], &y[0], n_features,
                      &w[0])

        # theta = y * sigmoid(-y * Xw) / alpha
        create_dual_pt(LOGREG, n_samples, alpha, &theta[0], &Xw[0], &y[0])
        norm_Xtheta = dual_norm(
            is_sparse, n_features, n_samples, &theta[0], X, X_data,
            X_indices, X_indptr, n_features, &all_features[0])

        if norm_Xtheta > 1.:
            tmp = 1. / norm_Xtheta
            fscal(&n_samples, &tmp, &theta[0], &inc)

        d_obj = dual(LOGREG, n_samples, alpha, 0., &theta[0], &y[0])
        gap = p_obj - d_obj
        gaps[t] = gap

        if t != 0 and use_accel:
            # do some epochs of CD to create an extrapolated dual point
            for k in range(K):
                cd_one_pass(w, is_sparse, X, X_data,
                            X_indices, X_indptr, y, alpha, Xw)
                fcopy(&n_samples, &Xw[0], &inc,
                      &last_K_Xw[k, 0], &inc)

            # TODO use function in utils
            for k in range(K - 1):
                for i in range(n_samples):
                    U[k, i] = last_K_Xw[k + 1, i] - last_K_Xw[k, i]

            for k in range(K - 1):
                for j in range(k, K - 1):
                    UUt[k, j] = fdot(&n_samples, &U[k, 0], &inc,
                                    &U[j, 0], &inc)
                    UUt[j, k] = UUt[k, j]

            for k in range(K - 1):
                onesK[k] = 1.

            fposv(char_U, &Kminus1, &one, &UUt[0, 0], &Kminus1,
                &onesK[0], &Kminus1, &info_dposv)

            if info_dposv != 0:
                for k in range(K - 2):
                    onesK[k] = 0
                onesK[K - 2] = 1

            sum_z = 0.
            for k in range(K - 1):
                sum_z += onesK[k]
            for k in range(K - 1):
                onesK[k] /= sum_z

            for i in range(n_samples):
                theta_acc[i] = 0.
            for k in range(K - 1):
                for i in range(n_samples):
                    theta_acc[i] += onesK[k] * last_K_Xw[k, i]
            for i in range(n_samples):
                theta_acc[i] = y[i] * sigmoid(- y[i] * theta_acc[i])

            tmp = 1. / alpha
            fscal(&n_samples, &tmp, &theta_acc[0], &inc)

            # do not forget to update exp_Xw
            for i in range(n_samples):
                exp_Xw[i] = exp(Xw[i])

            norm_Xtheta_acc = dual_norm(
                is_sparse, n_features, n_samples, &theta_acc[0], X, X_data,
                X_indices, X_indptr, n_features, &all_features[0])

            if norm_Xtheta_acc > 1.:
                tmp = 1. / norm_Xtheta_acc
                fscal(&n_samples, &tmp, &theta_acc[0], &inc)

            d_obj_acc = dual(LOGREG, n_samples, alpha, 0., &theta_acc[0], &y[0])
            if d_obj_acc > d_obj:
                fcopy(&n_samples, &theta_acc[0], &inc, &theta[0], &inc)
                gap = p_obj - d_obj_acc

        if verbose:
            print("############ Iteration %d #################" % t)
            print("Primal {:.10f}".format(p_obj))
            print("Dual {:.10f}, (gap: {:.2e})".format(d_obj, p_obj - d_obj))
            if use_accel:
                print("Acce {:.10f}, (gap: {:.2e})".format(
                    d_obj_acc, p_obj - d_obj_acc))

        if gap < tol:
            if verbose:
                print("Early exit, gap: %.2e < %.2e" % (gap, tol))
            break

        set_feature_prios(prios, is_sparse, n_features, n_samples, X,
                          X_data, X_indices, X_indptr, theta,
                          norms_X_col)

        if prune:
            if t == 0:
                ws_size = p0
            else:
                ws_size = 0
                for j in range(n_features):
                    if w[j] != 0:
                        prios[j] = -1
                        ws_size += 1
                ws_size = 2 * ws_size
                # ws_size += 1
        else:
            if t == 0:
                ws_size = p0
            else:
                #  ws_size = int(growth * ws_size)
                 ws_size *= 2

        if ws_size >= n_features:
            ws_size = n_features
            WS = all_features  # argpartition breaks otherwise
        else:
            WS = np.asarray(np.argpartition(prios, ws_size)[:ws_size])
            np.asarray(WS).sort()

        tol_inner = eps_inner * gap
        if verbose:
            print("Solving subproblem with %d constraints" % len(WS))

        PN_logreg(is_sparse, w, WS, X, X_data, X_indices, X_indptr, y,
                  alpha, tol_inner, Xw, exp_Xw, low_exp_Xw,
                  aux, is_positive_label, blitz_sc)

    return np.asarray(w), np.asarray(theta), np.asarray(gaps[:t + 1])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef int PN_logreg(
        bint is_sparse, floating[:] w, long[:] WS,
        floating[::1, :] X, floating[:] X_data, int[:] X_indices,
        int[:] X_indptr, floating[:] y, floating alpha,
        floating tol_inner, floating[:] Xw,
        floating[:] exp_Xw, floating[:] low_exp_Xw, floating[:] aux,
        int[:] is_positive_label, bint blitz_sc):

    cdef int n_samples = Xw.shape[0]
    cdef int ws_size = WS.shape[0]
    cdef int n_features = w.shape[0]

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef:
        int MAX_BACKTRACK_ITR = 10
        int MAX_PN_CD_ITR = 10
        int MIN_PN_CD_ITR = 2
    cdef floating PN_EPSILON_RATIO = 10.

    cdef floating[:] weights = np.zeros(n_samples, dtype=dtype)
    cdef floating[:] grad = np.zeros(n_samples, dtype=dtype)

    # solve a Lasso with other X and y (see paper)
    cdef floating[:] bias = np.zeros(ws_size, dtype=dtype)
    cdef floating[:] lc = np.zeros(ws_size, dtype=dtype)
    cdef floating[:] delta_w = np.zeros(ws_size, dtype=dtype)
    cdef floating[:] X_delta_w = np.zeros(n_samples, dtype=dtype)

    cdef floating[:] theta = np.zeros(n_samples, dtype=dtype)

    # for CD stopping criterion
    cdef bint first_pn_iteration = True
    cdef floating pn_grad_diff = 0.
    cdef floating approx_grad, actual_grad, sum_sq_hess_diff, pn_epsilon
    cdef floating[:] pn_grad_cache = np.zeros(ws_size, dtype=dtype)

    cdef int i, j, ind, max_cd_itr, cd_itr
    cdef floating prob

    cdef int start_ptr, end_ptr
    cdef floating  gap, p_obj, d_obj, norm_Xtheta, norm_Xaux
    cdef floating tmp, new_value, old_value, diff


    while True:
        # run prox newton iterations:
        for i in range(n_samples):
            prob = 1. / (1. + exp(y[i] * Xw[i]))
            weights[i] = prob * (1. - prob)
            grad[i] = - y[i] * prob

        for ind in range(ws_size):
            lc[ind] = wdot(Xw, weights, WS[ind], is_sparse, X, X_data,
                         X_indices, X_indptr, 1)
            bias[ind] = xj_dot(grad, WS[ind], is_sparse, X,
                             X_data, X_indices, X_indptr, n_features)

        if first_pn_iteration:
            # very weird: first cd iter, do only
            max_cd_itr = MIN_PN_CD_ITR
            pn_epsilon = 0
            first_pn_iteration = False
        else:
            max_cd_itr = MAX_PN_CD_ITR
            pn_epsilon = PN_EPSILON_RATIO * pn_grad_diff

        for ind in range(ws_size):
            delta_w[ind] = 0.
        for i in range(n_samples):
            X_delta_w[i] = 0
        for cd_itr in range(max_cd_itr):
            sum_sq_hess_diff = 0.

            for ind in range(ws_size):
                j = WS[ind]
                old_value = w[j] + delta_w[ind]
                tmp = wdot(X_delta_w, weights, j, is_sparse, X,
                           X_data, X_indices, X_indptr, jj=False)
                new_value = ST(old_value - (bias[ind] + tmp) / lc[ind],
                               alpha / lc[ind])

                diff = new_value - old_value
                if diff != 0:
                    sum_sq_hess_diff += lc[ind] ** 2 * diff ** 2
                    delta_w[ind] = new_value - w[j]
                    if is_sparse:
                        start_ptr, end_ptr = X_indptr[j], X_indptr[j + 1]
                        for i in range(start_ptr, end_ptr):
                            X_delta_w[X_indices[i]] += diff * X_data[i]
                    else:
                        for i in range(n_samples):
                            X_delta_w[i] += diff * X[i, j]
            if (sum_sq_hess_diff < pn_epsilon and
                    cd_itr + 1 >= MIN_PN_CD_ITR):
                break

        do_line_search(w, WS, delta_w, X_delta_w, Xw, alpha, is_sparse, X, X_data,
                       X_indices, X_indptr, MAX_BACKTRACK_ITR, y,
                       exp_Xw, low_exp_Xw, aux, is_positive_label)
        # aux is an up-to-date gradient (= - alpha * unscaled dual point)

        if blitz_sc:  # blitz stopping criterion for CD iter
            pn_grad_diff = 0.
            for ind in range(ws_size):
                j = WS[ind]
                actual_grad = xj_dot(
                    aux, j, is_sparse, X, X_data, X_indices, X_indptr, n_features)
                # TODO step_size taken into account?
                approx_grad = pn_grad_cache[ind] + wdot(
                    X_delta_w, weights, j, is_sparse, X, X_data, X_indices,
                    X_indptr, False)
                pn_grad_cache[ind] = actual_grad
                diff = approx_grad - actual_grad

                pn_grad_diff += diff ** 2

            norm_Xaux = 0.
            for ind in range(ws_size):
                tmp = fabs(pn_grad_cache[ind])
                if tmp > norm_Xaux:
                    norm_Xaux = tmp

        else:
            # rescale aux to create dual point
            norm_Xaux = dual_norm(
                is_sparse, n_features, n_samples, &aux[0], X,
                X_data, X_indices, X_indptr, ws_size, &WS[0])


        for i in range(n_samples):
            aux[i] /= - max(alpha, norm_Xaux)

        d_obj = dual(LOGREG, n_samples, alpha, 0, &aux[0], &y[0])
        p_obj = primal(LOGREG, alpha, n_samples, &Xw[0], &y[0],
                       n_features, &w[0])

        gap = p_obj - d_obj
        if gap < tol_inner:
            break


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void do_line_search(
        floating[:] w, long[:] WS, floating[:] delta_w,
        floating[:] X_delta_w, floating[:] Xw, floating alpha, bint is_sparse,
        floating[::1, :] X, floating[:] X_data,
        int[:] X_indices, int[:] X_indptr, int MAX_BACKTRACK_ITR,
        floating[:] y, floating[:] exp_Xw, floating[:] low_exp_Xw,
        floating[:] aux, int[:] is_positive_label) nogil:

    cdef int i, ind, backtrack_itr
    cdef floating deriv
    cdef floating step_size = 1.

    cdef int n_samples = y.shape[0]
    fcopy(&n_samples, &exp_Xw[0], &inc, &low_exp_Xw[0], &inc)

    for i in range(n_samples):
        exp_Xw[i] = exp(Xw[i] + X_delta_w[i])

    for backtrack_itr in range(MAX_BACKTRACK_ITR):
        compute_aux(aux, is_positive_label, exp_Xw)

        deriv = compute_derivative(
            w, WS, delta_w, X_delta_w, alpha, aux, step_size, y)

        if deriv < 1e-7:
            break
        else:
            step_size = step_size / 2.
        for i in range(n_samples):
            exp_Xw[i] = sqrt(exp_Xw[i] * low_exp_Xw[i])
    else:
        pass
        # TODO what do we do in this case?

    # a suitable step size is found, perform step:
    for ind in range(WS.shape[0]):
        w[WS[ind]] += step_size * delta_w[ind]
    for i in range(Xw.shape[0]):
        Xw[i] += step_size * X_delta_w[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef floating compute_derivative(
        floating[:] w, long[:] WS, floating[:] delta_w,
        floating[:] X_delta_w, floating alpha, floating[:] aux,
        floating step_size, floating[:] y) nogil:

    cdef int j
    cdef floating deriv_l1 = 0.
    cdef floating deriv_loss, wj
    cdef int n_samples = X_delta_w.shape[0]

    for j in range(WS.shape[0]):

        wj = w[WS[j]] + step_size * delta_w[j]
        if wj == 0.:
            deriv_l1 -= fabs(delta_w[j])
        else:
            deriv_l1 += wj / fabs(wj) * delta_w[j]

    deriv_loss = fdot(&n_samples, &X_delta_w[0], &inc, &aux[0], &inc)
    return deriv_loss + alpha * deriv_l1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void compute_aux(floating[:] aux, int[:] is_positive_label,
                       floating[:] exp_Xw) nogil:
    """-y / (1. + exp(y * Xw))"""
    cdef int i
    for i in range(is_positive_label.shape[0]):
        if is_positive_label[i]:
            aux[i] = -1 / (1. + exp_Xw[i])
        else:
            aux[i] = 1. - 1 / (1. + exp_Xw[i])



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef floating wdot(floating[:] v, floating[:] weights, int j,
        bint is_sparse, floating[::1, :] X, floating[:] X_data,
        int[:] X_indices, int[:] X_indptr, bint jj) nogil:
    """Weighted dot product between j-th column of X and v.

    Parameters:
    ----------
    jj: bool
        If true, v is ignored and dot product is between X[:, j] and X[:, j]
    """
    cdef floating tmp = 0
    cdef int start, end
    cdef int i

    if jj:
        if is_sparse:
            start, end = X_indptr[j], X_indptr[j + 1]
            for i in range(start, end):
                tmp += X_data[i] * X_data[i] * weights[X_indices[i]]
        else:
            for i in range(X.shape[0]):
                tmp += X[i, j] ** 2 * weights[i]
    else:
        if is_sparse:
            start, end = X_indptr[j], X_indptr[j + 1]
            for i in range(start, end):
                tmp += X_data[i] * v[X_indices[i]] * weights[X_indices[i]]
        else:
            for i in range(X.shape[0]):
                tmp += X[i, j] * v[i] * weights[i]
    return tmp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double xj_dot(floating[:] v, int j,  bint is_sparse,
       floating[::1, :] X, floating[:] X_data, int[:] X_indices,
       int[:] X_indptr, int n_features) nogil:
    """Dot product between j-th column of X and v."""
    cdef floating tmp = 0
    cdef int start, end
    cdef int i


    if is_sparse:
        start, end = X_indptr[j], X_indptr[j + 1]
        for i in range(start, end):
            tmp += X_data[i] * v[X_indices[i]]
    else:
        for i in range(n_features):
            tmp += X[i, j] * v[i]
    return tmp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void cd_one_pass(
        floating[:] w, bint is_sparse,
        floating[::1, :] X, floating[:] X_data,
        int[:] X_indices, int[:] X_indptr, floating[:] y,
        floating alpha, floating[:] Xw):
    """
    Do one pass of CD on non zero elements of w. Modifies w and Xw inplace
    """
    cdef int n_features = w.shape[0]
    cdef int n_samples = Xw.shape[0]

    cdef floating old_w_j, grad_j, lc_j, exp_yXw_i, tmp
    cdef int startptr, endptr
    cdef int i, j, ind

    for j in range(n_features):
        if not w[j]:
            continue
        old_w_j = w[j]
        grad_j = 0.

        if is_sparse:
            startptr = X_indptr[j]
            endptr = X_indptr[j + 1]

            for i in range(startptr, endptr):
                ind = X_indices[i]
                grad_j -= X_data[i] * y[ind] / (1. + exp(y[ind] * Xw[ind]))
        else:
            for i in range(n_samples):
                grad_j -= X[i, j] * y[i] / (1. + exp(y[i] * Xw[i]))

        lc_j = 0.

        if is_sparse:
            startptr = X_indptr[j]
            endptr = X_indptr[j + 1]

            for i in range(startptr, endptr):
                ind = X_indices[i]
                exp_yXw_i = exp(-y[ind] * Xw[ind])
                lc_j += X_data[i] ** 2 * exp_yXw_i / (1. + exp_yXw_i) ** 2
        else:
            for i in range(n_samples):
                exp_yXw_i = exp(- y[i] * Xw[i])
                lc_j += (X[i, j] ** 2 * exp_yXw_i / (1. + exp_yXw_i) ** 2)
        w[j] = ST(w[j] - grad_j / lc_j, alpha / lc_j)

        if old_w_j != w[j]:
            if is_sparse:
                startptr = X_indptr[j]
                endptr = X_indptr[j + 1]
                tmp = w[j] - old_w_j

                for i in range(startptr, endptr):
                    Xw[X_indices[i]] += tmp * X_data[i]

            else:
                for i in range(n_samples):
                    Xw[i] += (w[j] - old_w_j) * X[i, j]