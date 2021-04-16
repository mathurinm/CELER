# Author: Mathurin Massias <mathurin.massias@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Joseph Salmon <joseph.salmon@telecom-paristech.fr>
# License: BSD 3 clause

import numpy as np

from scipy import sparse
from numpy.linalg import norm
from sklearn.utils import check_array
from sklearn.linear_model._base import _preprocess_data

from .lasso_fast import celer
from .group_fast import celer_grp, dnorm_grp
from .cython_utils import compute_norms_X_col, compute_Xw
from .cython_utils import dnorm_l1 as dnorm_l1_cython
from .multitask_fast import celer_mtl
from .PN_logreg import newton_celer

LASSO = 0
LOGREG = 1
GRPLASSO = 2


def _log_weights(w, epsilon=0):
    reweights = np.full(w.shape[0], np.inf)
    supp = (w != 0)
    reweights[supp] = 1 / (np.abs(w[supp]) + epsilon)
    return reweights


def _sqrt_weights(w, epsilon=0):
    reweights = np.full(w.shape[0], np.inf)
    supp = (w != 0)
    reweights[supp] = 1. / (2 * np.sqrt(np.abs(w[supp])) + epsilon)
    return reweights


def celer_path(X, y, pb, eps=1e-3, n_alphas=100, alphas=None,
               coef_init=None, max_iter=20, max_epochs=50000,
               p0=10, verbose=0, tol=1e-6, prune=0, weights=None,
               n_reweightings=1, reweighting='log', groups=None,
               return_thetas=False, use_PN=False, X_offset=None, X_scale=None,
               return_n_iter=False,
               positive=False):
    r"""Compute optimization path with Celer as inner solver.

    With `n = len(y)` and `p = len(w)` the number of samples and features,
    the losses are:

    Lasso:

    .. math::

        \frac{||y - X w||_2^2}{2 n} + \alpha \sum_1^p weights_j |w_j|

    Logreg:

    .. math::

        \sum_{i=1}^n \text{log} \,(1 + e^{-y_i x_i^\top w}) + \alpha
        \sum_1^p weights_j |w_j|

    GroupLasso, with `G` the number of groups and `w_{[g]}` the subvector
    corresponding the group `g`:

    .. math::

        \frac{||y - X w||_2^2}{2 n} + \alpha \sum_1^G weights_g ||w_{[g]}||_2

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data or column
        sparse format (CSC) to avoid unnecessary memory duplication.

    y : ndarray, shape (n_samples,)
        Target values

    pb : "lasso" | "logreg" | "grouplasso"
        Optimization problem to solve.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min = 1e-3 * alpha_max``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    coef_init : ndarray, shape (n_features,) | None, optional, (default=None)
        Initial value of coefficients. If None, np.zeros(n_features) is used.

    max_iter : int, optional
        The maximum number of iterations (definition of working set and
        resolution of problem restricted to features in working set)

    max_epochs : int, optional
        Maximum number of coordinate descent epochs on each subproblem.

    p0 : int, optional
        First working set size.

    verbose : bool or integer, optional
        Amount of verbosity. 0/False is silent

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    prune : bool, optional
        Whether or not to use pruning when growing working sets.

    weights : ndarray, shape (n_features,) or (n_groups,), optional
        Feature/group weights used in the penalty. Default to array of ones.
        Features with weights equal to np.inf are ignored.

    n_reweightings : int, optional (default=1)
        If > 1, for each :math:`\alpha`, `n_reweightings` weighted Lassos are
        solved, taking the inverse magnitude of the previous coefficient as
        weights.
        This procedure amounts to a majorization-minimization approach to
        handle L0.5 regularization, and should reduce the coefficient bias.

    reweighting : string or callable (default='log')
        Reweighting scheme if `n_reweightings` > 1.
        String supported are 'log' and 'sqrt', corresponding to log and sqrt
        penalties.
        If a callable is passed, it is given the previous coefficients
        and must return an array of corresponding new weights.

    groups : int or list of ints or list of list of ints, optional
        Used for the group Lasso only. See the documentation of the
        GroupLasso class.

    return_thetas : bool, optional
        If True, dual variables along the path are returned.

    use_PN : bool, optional
        If pb == "logreg", use ProxNewton solver instead of coordinate
        descent.

    X_offset : np.array, shape (n_features,), optional
        Used to center sparse X without breaking sparsity. Mean of each column.
        See sklearn.linear_model.base._preprocess_data().

    X_scale : np.array, shape (n_features,), optional
        Used to scale centered sparse X without breaking sparsity. Norm of each
        centered column. See sklearn.linear_model.base._preprocess_data().

    return_n_iter : bool, optional
        If True, number of iterations along the path are returned.

    positive : bool, optional (default=False)
        If True and pb == "lasso", forces the coefficients to be positive.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : array, shape (n_alphas,)
        Duality gaps returned by the solver along the path.

    thetas : array, shape (n_alphas, n_samples)
        The dual variables along the path.
        (Is returned only when ``return_thetas`` is set to True).
    """

    if pb.lower() not in ("lasso", "logreg", "grouplasso"):
        raise ValueError("Unsupported problem %s" % pb)
    if pb.lower() == "lasso":
        pb = LASSO
    elif pb.lower() == "logreg":
        pb = LOGREG
        if set(y) - set([-1.0, 1.0]):
            raise ValueError(
                "y must contain only -1. or 1 values. Got %s " % (set(y)))
    elif pb.lower() == "grouplasso":
        pb = GRPLASSO
        if groups is None:
            raise ValueError(
                "Groups must be specified for the group lasso problem.")
        grp_ptr, grp_indices = _grp_converter(groups, X.shape[1])
        n_groups = len(grp_ptr) - 1
    else:
        raise ValueError("Unsupported problem: %s" % pb)

    is_sparse = sparse.issparse(X)

    X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                    order='F', copy=False, accept_large_sparse=False)
    y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                    ensure_2d=False)

    n_samples, n_features = X.shape

    if X_offset is not None:
        # As sparse matrices are not actually centered we need this
        # to be passed to the CD solver.
        X_sparse_scaling = X_offset / X_scale
        X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
    else:
        X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    X_dense, X_data, X_indices, X_indptr = _sparse_and_dense(X)

    if weights is None:
        weights = np.ones(n_groups) if pb == GRPLASSO else np.ones(n_features)
        weights = weights.astype(X.dtype)
    elif (weights <= 0).any():
        raise ValueError("0 or negative weights are not supported.")

    if isinstance(reweighting, str):
        if reweighting == 'log':
            reweighting = _log_weights
        elif reweighting == 'sqrt':
            reweighting = _sqrt_weights
        else:
            raise ValueError("Unsupported reweighting scheme %s" % reweighting)

    if alphas is None:
        if pb == LASSO:
            alpha_max = dnorm_l1(X, y, weights, X_sparse_scaling,
                                 positive) / n_samples
        elif pb == LOGREG:
            alpha_max = dnorm_l1(X, y, weights, X_sparse_scaling,
                                 positive) / 2
        elif pb == GRPLASSO:
            # TODO compute it with dscal to handle centering sparse
            alpha_max = 0
            for g in range(n_groups):
                X_g = X[:, grp_indices[grp_ptr[g]:grp_ptr[g + 1]]]
                alpha_max = max(alpha_max, norm(X_g.T @ y, ord=2))
            alpha_max /= n_samples

        alphas = alpha_max * np.geomspace(1, eps, n_alphas,
                                          dtype=X.dtype)
    else:
        alphas = np.sort(alphas)[::-1]

    n_alphas = len(alphas)

    coefs = np.zeros((n_features, n_alphas), order='F', dtype=X.dtype)
    thetas = np.zeros((n_alphas, n_samples), dtype=X.dtype)
    dual_gaps = np.zeros(n_alphas)

    if return_n_iter:
        n_iters = np.zeros(n_alphas, dtype=int)

    if pb == GRPLASSO:
        # TODO this must be included in compute_norm_Xcols when centering
        norms_X_grp = np.zeros(n_groups, dtype=X_dense.dtype)
        for g in range(n_groups):
            X_g = X[:, grp_indices[grp_ptr[g]:grp_ptr[g + 1]]]
            if is_sparse:
                gram = (X_g.T @ X_g).todense()
                # handle centering:
                for j1 in range(grp_ptr[g], grp_ptr[g + 1]):
                    for j2 in range(grp_ptr[g], grp_ptr[g + 1]):
                        gram[j1 - grp_ptr[g], j2 - grp_ptr[g]] += \
                            X_sparse_scaling[j1] * \
                            X_sparse_scaling[j2] * n_samples - \
                            X_sparse_scaling[j1] * \
                            X_data[X_indptr[j2]:X_indptr[j2+1]].sum() - \
                            X_sparse_scaling[j2] * \
                            X_data[X_indptr[j1]:X_indptr[j1+1]].sum()

                norms_X_grp[g] = np.sqrt(norm(gram, ord=2))
            else:
                norms_X_grp[g] = norm(X_g, ord=2)
    else:
        # TODO harmonize names
        norms_X_col = np.zeros(n_features, dtype=X_dense.dtype)
        compute_norms_X_col(
            is_sparse, norms_X_col, n_samples, X_dense, X_data,
            X_indices, X_indptr, X_sparse_scaling)

    for reweight_iter in range(n_reweightings):
        # handle reweighting in a clever way: we must loop over reweighting
        # then over alphas, in order to use warm start efficiently.
        for t in range(n_alphas):
            # do not skip alphas[0], it is not always alpha_max
            alpha = alphas[t]
            if verbose:
                to_print = "##### Computing alpha %d/%d" % (t + 1, n_alphas)
                print("#" * len(to_print))
                print(to_print)
                print("#" * len(to_print))
            if t > 0:
                w = coefs[:, t - 1].copy()
                # theta = thetas[t - 1].copy()
                p0 = max(len(np.where(w != 0)[0]), 1)
            else:
                if coef_init is not None:
                    w = coef_init.copy()
                    p0 = max((w != 0.).sum(), p0)
                    # y - Xw for Lasso, Xw for Logreg:
                    Xw = np.zeros(n_samples, dtype=X.dtype)
                    compute_Xw(
                        is_sparse, pb, Xw, w, y, X_sparse_scaling.any(),
                        X_dense, X_data, X_indices, X_indptr, X_sparse_scaling)
                else:
                    w = np.zeros(n_features, dtype=X.dtype)
                    Xw = np.zeros(
                        n_samples, X.dtype) if pb == LOGREG else y.copy()

                # # different link eqs and normalization scal for dual point:
                # if pb in (LASSO, LOGREG):
                #     if pb == LASSO:
                #         theta = Xw.copy()
                #     elif pb == LOGREG:
                #         theta = y / (1 + np .exp(y * Xw)) / alpha
                #     scal = dnorm_l1(X, theta, weights, X_sparse_scaling,
                #                     positive)
                # elif pb == GRPLASSO:
                #     theta = Xw.copy()
                #     scal = dnorm_grp(
                #         is_sparse, theta, grp_ptr, grp_indices, X_dense,
                #         X_data, X_indices, X_indptr, X_sparse_scaling,
                #         len(grp_ptr) - 1, np.zeros(1, dtype=np.int32),
                #         X_sparse_scaling.any())
                # theta /= scal
            theta = np.zeros(n_samples, dtype=X.dtype)

            # celer modifies w, Xw, and theta in place:
            if pb == GRPLASSO:
                sol = celer_grp(
                    is_sparse, LASSO, X_dense, grp_indices, grp_ptr, X_data,
                    X_indices,
                    X_indptr, X_sparse_scaling, y, alpha, w, Xw, theta,
                    norms_X_grp, tol, max_iter, max_epochs, p0=p0,
                    prune=prune, verbose=verbose)
            elif pb == LASSO or (pb == LOGREG and not use_PN):
                if reweight_iter > 0:
                    reweights = reweighting(coefs[:, t])
                    reweights *= weights
                else:
                    reweights = weights.copy()
                sol = celer(
                    is_sparse, pb, X_dense, X_data, X_indices, X_indptr,
                    X_sparse_scaling, y, alpha, w, Xw, theta, norms_X_col,
                    reweights, max_iter, max_epochs, tol=tol, p0=p0,
                    verbose=verbose, prune=prune, positive=positive)

            else:  # pb == LOGREG and use_PN
                sol = newton_celer(
                    is_sparse, X_dense, X_data, X_indices, X_indptr, y, alpha,
                    w, max_iter, tol=tol, p0=p0, verbose=verbose, prune=prune)

            coefs[:, t], thetas[t], dual_gaps[t] = sol[0], sol[1], sol[2][-1]
            if return_n_iter:
                n_iters[t] = len(sol[2])  # meaningless for reweightings > 0

    results = alphas, coefs, dual_gaps
    if return_thetas:
        results += (thetas,)
    if return_n_iter:
        results += (n_iters,)

    return results


def _sparse_and_dense(X):
    if sparse.issparse(X):
        X_dense = np.empty([1, 1], order='F', dtype=X.data.dtype)
        X_data = X.data
        X_indptr = X.indptr
        X_indices = X.indices
    else:
        X_dense = X
        X_data = np.empty([1], dtype=X.dtype)
        X_indices = np.empty([1], dtype=np.int32)
        X_indptr = np.empty([1], dtype=np.int32)
    return X_dense, X_data, X_indices, X_indptr


def dnorm_l1(X, theta, weights, X_sparse_scaling, positive):
    """Theta should be centered."""
    X_dense, X_data, X_indices, X_indptr = _sparse_and_dense(X)
    skip = np.zeros(X.shape[1], dtype=np.int32)
    scal = dnorm_l1_cython(
        sparse.issparse(X), theta, X_dense, X_data, X_indices, X_indptr,
        skip, X_sparse_scaling, weights, X_sparse_scaling.any(), positive)
    return scal


def _alpha_max_grp(X, y, groups, center=False, normalize=False):
    """This costly function (copies X) should only be used for debug."""
    grp_ptr, grp_indices = _grp_converter(groups, X.shape[1])
    X, y, X_offset, _, X_scale = _preprocess_data(
        X, y, center, normalize, copy=True)

    X_mean = X_offset / X_scale
    X_dense, X_data, X_indices, X_indptr = _sparse_and_dense(X)
    alpha_max = dnorm_grp(
        sparse.issparse(X), y, grp_ptr, grp_indices, X_dense, X_data,
        X_indices, X_indptr, X_mean, len(grp_ptr) - 1,
        np.zeros(1, dtype=np.int32), X_mean.any()) / len(y)

    return alpha_max


def _grp_converter(groups, n_features):
    if isinstance(groups, int):
        grp_size = groups
        if n_features % grp_size != 0:
            raise ValueError("n_features (%d) is not a multiple of the desired"
                             " group size (%d)" % (n_features, grp_size))
        n_groups = n_features // grp_size
        grp_ptr = grp_size * np.arange(n_groups + 1)
        grp_indices = np.arange(n_features)
    elif isinstance(groups, list) and isinstance(groups[0], int):
        grp_indices = np.arange(n_features).astype(np.int32)
        grp_ptr = np.cumsum(np.hstack([[0], groups]))
    elif isinstance(groups, list) and isinstance(groups[0], list):
        grp_sizes = np.array([len(ls) for ls in groups])
        grp_ptr = np.cumsum(np.hstack([[0], grp_sizes]))
        grp_indices = np.array([idx for grp in groups for idx in grp])
    else:
        raise ValueError("Unsupported group format.")
    return grp_ptr.astype(np.int32), grp_indices.astype(np.int32)


def mtl_path(
        X, Y, eps=1e-2, n_alphas=100, alphas=None, max_iter=100,
        max_epochs=50_000, p0=10, verbose=0, tol=1e-6,
        prune=True, return_thetas=False, coef_init=None):
    X = check_array(X, "csc", dtype=[
        np.float64, np.float32], order="F", copy=False)
    Y = check_array(Y, "csc", dtype=[
        np.float64, np.float32], order="F", copy=False)
    n_samples, n_features = X.shape
    n_tasks = Y.shape[1]
    if alphas is None:
        alpha_max = np.max(norm(X.T @ Y, ord=2, axis=1)) / n_samples
        alphas = alpha_max * \
            np.geomspace(1, eps, n_alphas, dtype=X.dtype)
    else:
        alphas = np.sort(alphas)[::-1]

    n_alphas = len(alphas)

    coefs = np.zeros((n_features, n_tasks, n_alphas), order="F",
                     dtype=X.dtype)

    thetas = np.zeros((n_alphas, n_samples, n_tasks), dtype=X.dtype)
    gaps = np.zeros(n_alphas)

    norms_X_col = np.linalg.norm(X, axis=0)
    Y = np.asfortranarray(Y)
    R = Y.copy(order='F')
    theta = np.zeros_like(Y, order='F')

    # do not skip alphas[0], it is not always alpha_max
    for t in range(n_alphas):
        alpha = alphas[t]

        if verbose:
            msg = "##### Computing alpha %d/%d" % (t + 1, n_alphas)
            print("#" * len(msg))
            print(msg)
            print("#" * len(msg))
        if t > 0:
            W = coefs[:, :, t - 1].copy()
            p_t = max(len(np.where(W[:, 0] != 0)[0]), p0)
        else:
            if coef_init is not None:
                W = coef_init.T
                R = np.asfortranarray(Y - X @ W)
                p_t = max(len(np.where(W[:, 0] != 0)[0]), p0)
            else:
                W = np.zeros((n_features, n_tasks), dtype=X.dtype)
                p_t = 10

        sol = celer_mtl(
            X, Y, alpha, W, R, theta, norms_X_col, p0=p_t, tol=tol,
            prune=prune, max_iter=max_iter, max_epochs=max_epochs,
            verbose=verbose, use_accel=True)

        coefs[:, :, t], thetas[t], gaps[t] = sol[0], sol[1], sol[2]

    coefs = np.swapaxes(coefs, 0, 1).copy('F')

    if return_thetas:
        return alphas, coefs, gaps, thetas

    return alphas, coefs, gaps
