# flake8: noqa F401
import inspect
import numbers

import numpy as np

from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.linear_model import (Lasso as Lasso_sklearn,
                                  LogisticRegression as LogReg_sklearn,
                                  MultiTaskLasso as MultiTaskLasso_sklearn)
from sklearn.linear_model._coordinate_descent import LinearModelCV
from sklearn.linear_model._base import _preprocess_data
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

from .homotopy import celer_path, mtl_path


class Lasso(Lasso_sklearn):
    r"""
    Lasso scikit-learn estimator based on Celer solver

    With :math:`n` the number of samples, the optimization objective for Lasso
    is::

        ||y - X w||^2_2 / (2 n) + alpha * \sum_j weights_j |w_j|

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square.
        For numerical reasons, using ``alpha = 0`` with the
        ``Lasso`` object is not advised.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization: the solver runs until the
        duality gap is smaller than ``tol * norm(y) ** 2 / len(y)`` or the
        maximum number of iteration is reached.

    prune : bool, optional
        Whether or not to use pruning when growing working sets.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    weights : array, shape (n_features,), optional (default=None)
        Strictly positive weights used in the L1 penalty part of the Lasso
        objective. If None, weights equal to 1 are used.

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional (default=False)
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    Examples
    --------
    >>> from celer import Lasso
    >>> clf = Lasso(alpha=0.1)
    >>> clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    Lasso(alpha=0.1, max_epochs=50000, max_iter=100,
    p0=10, prune=0, tol=1e-06, verbose=0)
    >>> print(clf.coef_)
    [0.85 0.  ]
    >>> print(clf.intercept_)
    0.15

    See also
    --------
    celer_path
    LassoCV

    References
    ----------
    .. [1] M. Massias, A. Gramfort, J. Salmon
      "Celer: a Fast Solver for the Lasso wit Dual Extrapolation", ICML 2018,
      http://proceedings.mlr.press/v80/massias18a.html
    """

    def __init__(self, alpha=1., max_iter=100, max_epochs=50000, p0=10,
                 verbose=0, tol=1e-4, prune=True, fit_intercept=True,
                 weights=None, normalize=False, warm_start=False,
                 positive=False):
        super(Lasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.positive = positive
        self.weights = weights

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute Lasso path with Celer."""
        results = celer_path(
            X, y, "lasso", alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            max_epochs=self.max_epochs, p0=self.p0, verbose=self.verbose,
            tol=self.tol, prune=self.prune, weights=self.weights,
            positive=self.positive, X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))

        return results


class LassoCV(RegressorMixin, LinearModelCV):
    r"""
    LassoCV scikit-learn estimator based on Celer solver

    The best model is selected by cross-validation.

    With :math:`n` the number of samples, the optimization objective for Lasso
    is::

        ||y - X w||^2_2 / (2 n) + alpha * \sum_j weights_j |w_j|


    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path.

    alphas : numpy array, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    fit_intercept : boolean, default True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    tol : float, optional
        Stopping criterion for the optimization: the solver runs until the
        duality gap is smaller than ``tol * norm(y) ** 2 / len(y)`` or the
        maximum number of iteration is reached.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, sklearn `KFold` is used.

    verbose : bool or integer
        Amount of verbosity.

    max_epochs : int, optional (default=50000)
        Maximum number of coordinate descent epochs when solving a subproblem.

    p0 : int, optional (default=10)
        Number of features in the first working set.

    prune : bool, optional (default=False)
        Whether to use pruning when growing the working sets.

    precompute : ignored parameter, kept for sklearn compatibility.

    positive : bool, optional (default=False)
        When set to True, forces the coefficients to be positive.

    n_jobs : int or None, optional (default=None)
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    intercept_ : float
        independent term in decision function.

    mse_path_ : array, shape (n_alphas, n_folds)
        mean square error for the test set on each fold, varying alpha

    alphas_ : numpy array, shape (n_alphas,)
        The grid of alphas used for fitting

    dual_gap_ : ndarray, shape ()
        The dual gap at the end of the optimization for the optimal alpha
        (``alpha_``).

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    See also
    --------
    celer_path
    Lasso
    """

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, max_iter=100,
                 tol=1e-4, cv=None, verbose=0, max_epochs=50000, p0=10,
                 prune=True, precompute='auto', positive=False, n_jobs=None):
        super(LassoCV, self).__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas, max_iter=max_iter,
            tol=tol, cv=cv, fit_intercept=fit_intercept, normalize=normalize,
            verbose=verbose, n_jobs=n_jobs)
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.positive = positive

    def path(self, X, y, alphas, coef_init=None, **kwargs):
        """Compute Lasso path with Celer."""
        alphas, coefs, dual_gaps = celer_path(
            X, y, "lasso", alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, max_epochs=self.max_epochs,
            p0=self.p0, verbose=self.verbose, tol=self.tol, prune=self.prune,
            positive=self.positive, X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))
        return alphas, coefs, dual_gaps

    def _get_estimator(self):
        return Lasso()

    def _is_multitask(self):
        return False

    def _more_tags(self):
        return {'multioutput': False}


class AdaptiveLasso(Lasso):
    """
    AdaptiveLasso scikit-learn estimator based on Celer solver.

    With :math:`n` the number of samples, the optimization objective for Lasso
    is::

        ||y - X w||^2_2 / (2 n) + alpha * \sum_j weights_j |w_j|

    The Adaptive Lasso solves a sequence of ``n_reweightings`` Lassos, with
    weights equal to the inverse magnitude of the coefficients at previous
    iteration::

        w^{it} = argmin ||y - X w||^2_2 / (2 n) + alpha * \sum_j weights_j \
w_j| / |w_j^{it - 1}|

    Features which had a zero coef at previous iterations have an infinite
    weight and are thus excluded from the model.


    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.

    n_reweightings : int, optional (default=5)
        Number of reweightings performed, ie number of Lasso problems solved.

    reweighting : string or callable (default='log')
        Reweighting scheme if `n_reweightings` > 1.
        String supported are 'log' and 'sqrt', corresponding to log and sqrt
        penalties.
        If a callable is passed, it is given the previous coefficients
        and must return an array of corresponding new weights.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    prune : bool, optional
        Whether or not to use pruning when growing working sets.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    weights : array, shape (n_features,), optional (default=None)
        Weights used in the L1 penalty part of the Lasso objective.
        If None, weights equal to 1 are used.

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional (default=False)
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    See also
    --------
    celer_path
    LassoCV

    References
    ----------
    .. [1] M. Massias, A. Gramfort, J. Salmon
      "Celer: a Fast Solver for the Lasso wit Dual Extrapolation", ICML 2018,
      http://proceedings.mlr.press/v80/massias18a.html
    """

    def __init__(self, alpha=1., n_reweightings=5, reweighting='log',
                 max_iter=100, max_epochs=50000, p0=10, verbose=0, tol=1e-4,
                 prune=True, fit_intercept=True, weights=None, normalize=False,
                 warm_start=False, positive=False):
        super(AdaptiveLasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start, verbose=verbose, max_epochs=max_epochs,
            p0=p0, prune=prune, positive=positive, weights=weights)
        self.n_reweightings = n_reweightings
        self.reweighting = reweighting

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute Lasso path with Celer."""
        results = celer_path(
            X, y, "lasso", alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            max_epochs=self.max_epochs, p0=self.p0, verbose=self.verbose,
            tol=self.tol, prune=self.prune, weights=self.weights,
            n_reweightings=self.n_reweightings, reweighting=self.reweighting,
            positive=self.positive, X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))

        return results


class AdaptiveLassoCV(LassoCV):
    r"""
    AdaptiveLassoCV scikit-learn estimator based on Celer solver

    The best model is selected by cross-validation.

    The AdaptiveLasso solves a sequence of weighted Lassos::

        (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * sum_j weights_j |w_j|

    Parameters
    ----------
    n_reweightings : int, optional (default=5)
        Number of reweightings performed for each alpha (ie number of Lasso
        problems solved).

    reweighting : string or callable (default='log')
        Reweighting scheme if `n_reweightings` > 1.
        String supported are 'log' and 'sqrt', corresponding to log and sqrt
        penalties.
        If a callable is passed, it is given the previous coefficients
        and must return an array of corresponding new weights.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path.

    alphas : numpy array, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    fit_intercept : boolean, default True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    tol : float, optional
        The tolerance for the optimization: the solver runs until the duality
        gap is smaller than ``tol`` or the maximum number of iteration is
        reached.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, sklearn `KFold` is used.

    verbose : bool or integer
        Amount of verbosity.

    max_epochs : int, optional (default=50000)
        Maximum number of coordinate descent epochs when solving a subproblem.

    p0 : int, optional (default=10)
        Number of features in the first working set.

    prune : bool, optional (default=False)
        Whether to use pruning when growing the working sets.

    precompute : ignored parameter, kept for sklearn compatibility.

    positive : bool, optional (default=False)
        When set to True, forces the coefficients to be positive.

    n_jobs : int or None, optional (default=None)
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    coef_ : array, shape (n_features,)
        Parameter vector (w in the cost function formula)

    intercept_ : float
        Independent term in decision function.

    mse_path_ : array, shape (n_alphas, n_folds)
        Mean square error for the test set on each fold, varying alpha

    alphas_ : numpy array, shape (n_alphas,)
        The grid of alphas used for fitting

    dual_gap_ : ndarray, shape ()
        The dual gap at the end of the optimization for the optimal alpha
        (``alpha_``).

    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    See also
    --------
    celer_path
    Lasso
    """

    def __init__(self, n_reweightings=5, reweighting='log', eps=1e-3,
                 n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, max_iter=100,
                 tol=1e-4, cv=None, verbose=0, max_epochs=50000, p0=10,
                 prune=True, precompute='auto', positive=False, n_jobs=None):
        super(AdaptiveLassoCV, self).__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas, max_iter=max_iter,
            tol=tol, cv=cv, fit_intercept=fit_intercept, normalize=normalize,
            verbose=verbose, n_jobs=n_jobs, max_epochs=max_epochs, p0=p0,
            prune=prune, positive=positive)
        self.n_reweightings = n_reweightings
        self.reweighting = reweighting

    def path(self, X, y, alphas, coef_init=None, **kwargs):
        """Compute Lasso path with Celer."""
        alphas, coefs, dual_gaps = celer_path(
            X, y, "lasso", alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, max_epochs=self.max_epochs,
            p0=self.p0, verbose=self.verbose, tol=self.tol, prune=self.prune,
            n_reweightings=self.n_reweightings, reweighting=self.reweighting,
            positive=self.positive,
            X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))
        return alphas, coefs, dual_gaps

    def _get_estimator(self):
        return AdaptiveLasso()

    def _is_multitask(self):
        return False

    def _more_tags(self):
        return {'multioutput': False}


class MultiTaskLasso(MultiTaskLasso_sklearn):
    r"""
    MultiTaskLasso scikit-learn estimator based on Celer solver

    The optimization objective for MultiTaskLasso is::

        (1 / (2 * n_samples)) * ||y - X W||^2_2 + alpha * ||W||_{21}

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square.
        For numerical reasons, using ``alpha = 0`` with the
        ``Lasso`` object is not advised.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization: the solver runs until the
        duality gap is smaller than ``tol * norm(y) ** 2 / len(y)`` or the
        maximum number of iteration is reached.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    See also
    --------
    celer_path
    MultiTaskLassoCV

    References
    ----------
    .. [1] M. Massias, A. Gramfort, J. Salmon
      "Celer: a Fast Solver for the Lasso wit Dual Extrapolation", ICML 2018,
      http://proceedings.mlr.press/v80/massias18a.html
    """

    def __init__(self, alpha=1., max_iter=100,
                 max_epochs=50000, p0=10, verbose=0, tol=1e-4, prune=0,
                 fit_intercept=True, normalize=False, warm_start=False):
        super().__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune

    def fit(self, X, y):
        """Fit MultiTaskLasso model with Celer"""
        # Need to validate separately here.
        # We can't pass multi_ouput=True because that would allow y to be csr.
        check_X_params = dict(dtype=[np.float64, np.float32], order='F',
                              copy=self.copy_X and self.fit_intercept)
        check_y_params = dict(ensure_2d=False, order='F')
        X, y = self._validate_data(X, y, validate_separately=(check_X_params,
                                                              check_y_params))
        y = y.astype(X.dtype)

        if y.ndim == 1:
            raise ValueError("For mono-task outputs, use Lasso")

        n_samples = X.shape[0]

        if n_samples != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (n_samples, y.shape[0]))

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, self.fit_intercept, self.normalize, copy=False)

        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = None

        _, coefs, dual_gaps = mtl_path(
            X, y, alphas=[self.alpha], coef_init=self.coef_,
            max_iter=self.max_iter, max_epochs=self.max_epochs, p0=self.p0,
            verbose=self.verbose, tol=self.tol, prune=self.prune)

        self.coef_, self.dual_gap_ = coefs[..., 0], dual_gaps[-1]
        self.n_iter_ = len(dual_gaps)
        self._set_intercept(X_offset, y_offset, X_scale)

        return self


class MultiTaskLassoCV(RegressorMixin, LinearModelCV):
    r"""
    MultiTaskLassoCV scikit-learn estimator based on Celer solver

    The best model is selected by cross-validation.

    The optimization objective for Multi-task Lasso is::

        (1 / (2 * n_samples)) * ||y - X W||^2_2 + alpha * ||W||_{21}

    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path.

    alphas : numpy array, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    fit_intercept : boolean, default True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    tol : float, optional
        Stopping criterion for the optimization: the solver runs until the
        duality gap is smaller than ``tol * norm(y) ** 2 / len(y)`` or the
        maximum number of iteration is reached.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, sklearn `KFold` is used.

    verbose : bool or integer
        Amount of verbosity.

    max_epochs : int, optional (default=50000)
        Maximum number of coordinate descent epochs when solving a subproblem.

    p0 : int, optional (default=10)
        Number of features in the first working set.

    prune : bool, optional (default=True)
        Whether to use pruning when growing the working sets.

    precompute : ignored parameter, kept for sklearn compatibility.

    n_jobs : int
        to run CV in parallel.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    coef_ : array, shape (n_features, n_outputs)
        parameter vector (w in the cost function formula)

    intercept_ : array, shape (n_outputs,)
        independent term in decision function.

    mse_path_ : array, shape (n_alphas, n_folds)
        mean square error for the test set on each fold, varying alpha

    alphas_ : numpy array, shape (n_alphas,)
        The grid of alphas used for fitting

    dual_gap_ : ndarray, shape (n_alphas,)
        The dual gap at the end of the optimization for the optimal alpha
        (``alpha_``).

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    See also
    --------
    celer_path
    MultiTaskLasso
    """

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, max_iter=100,
                 tol=1e-4, cv=None, verbose=0,
                 max_epochs=50000, p0=10, prune=True, precompute='auto',
                 n_jobs=1):
        super().__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas, max_iter=max_iter,
            tol=tol, cv=cv, fit_intercept=fit_intercept, normalize=normalize,
            verbose=verbose, n_jobs=n_jobs)
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune

    def path(self, X, y, alphas, coef_init=None, **kwargs):
        """Compute Lasso path with Celer."""

        alphas, coefs, dual_gaps = mtl_path(
            X, y, alphas=alphas, coef_init=coef_init, max_iter=self.max_iter,
            max_epochs=self.max_epochs, p0=self.p0,
            verbose=self.verbose, tol=self.tol, prune=self.prune)

        return alphas, coefs, dual_gaps

    def _get_estimator(self):
        return MultiTaskLasso()

    def _is_multitask(self):
        return True

    def _more_tags(self):
        return {'multioutput_only': True}


class LogisticRegression(LogReg_sklearn):
    r"""
    Sparse Logistic regression scikit-learn estimator based on Celer solver.

    The optimization objective for sparse Logistic regression is::

        \sum_1^n_samples log(1 + e^{-y_i x_i^T w}) + 1. / C * ||w||_1

    The solvers use a working set strategy. To solve problems restricted to a
    subset of features, Celer uses coordinate descent while PN-Celer uses
    a Prox-Newton strategy (detailed in [1], Sec 5.2).

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.

    penalty : 'l1'.
        Other penalties are not supported.

    solver : "celer" | "celer-pn", default="celer-pn"
        Algorithm to use in the optimization problem.

        - celer-pn uses working sets and prox-Newton solver on the working set.
        - celer uses working sets and coordinate descent

    tol : float, optional
        Stopping criterion for the optimization: the solver runs until the
        duality gap is smaller than ``tol * len(y) * log(2)`` or the
        maximum number of iteration is reached.

    fit_intercept : bool, optional (default=False)
        Whether or not to fit an intercept. Currently True is not supported.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    verbose : bool or integer
        Amount of verbosity.

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Only False is supported so far.


    Attributes
    ----------

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.

    intercept_ :  ndarray of shape (1,) or (n_classes,)
        constant term in decision function. Not handled yet.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    Examples
    --------
    >>> from celer import LogisticRegression
    >>> clf = LogisticRegression(C=1.)
    >>> clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 1])
    LogisticRegression(C=1.0, penalty='l1', tol=0.0001, fit_intercept=False,
    max_iter=50, verbose=False, max_epochs=50000, p0=10, warm_start=False)

    >>> print(clf.coef_)
    [[0.4001237  0.01949392]]

    See also
    --------
    celer_path

    References
    ----------
    .. [1] M. Massias, S. Vaiter, A. Gramfort, J. Salmon
       "Dual Extrapolation for Sparse Generalized Linear Models", JMLR 2020,
       https://arxiv.org/abs/1907.05830
    """

    def __init__(self, C=1., penalty='l1', solver="celer-pn", tol=1e-4,
                 fit_intercept=False, max_iter=50, verbose=False,
                 max_epochs=50000, p0=10, warm_start=False):
        super(LogisticRegression, self).__init__(
            tol=tol, C=C)

        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.max_iter = max_iter
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.solver = solver

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        # TODO handle normalization, centering
        # TODO intercept
        if self.fit_intercept:
            raise NotImplementedError(
                "Fitting an intercept is not implement yet")
        # TODO support warm start
        if self.penalty != 'l1':
            raise NotImplementedError(
                'Only L1 penalty is supported, got %s' % self.penalty)

        if not isinstance(self.C, numbers.Number) or self.C <= 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        # below are copy pasted excerpts from sklearn.linear_model._logistic
        X, y = check_X_y(X, y, accept_sparse='csr', order="C")
        check_classification_targets(y)
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        self.classes_ = enc.classes_
        n_classes = len(enc.classes_)

        if n_classes <= 2:
            coefs = self.path(
                X, 2 * y_ind - 1, np.array([self.C]), solver=self.solver)[0]
            self.coef_ = coefs.T  # must be [1, n_features]
            self.intercept_ = 0
        else:
            self.coef_ = np.empty([n_classes, X.shape[1]])
            self.intercept_ = 0.
            multiclass = OneVsRestClassifier(self).fit(X, y)
            self.coef_ = multiclass.coef_

        return self

    def path(self, X, y, Cs, solver, coef_init=None, **kwargs):
        """
        Compute sparse Logistic Regression path with Celer-PN.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Cs : ndarray
            Values of regularization strenghts for which solutions are
            computed

        solver : "celer-pn" | "celer"
            Algorithm used to solve the optimization problem.

        coef_init : array, shape (n_features,), optional
            Initial value of the coefficients.

        Returns
        -------
        coefs_ : array, shape (len(Cs), n_features)
            Computed coefficients for each value in Cs.

        dual_gaps : array, shape (len(Cs),)
            Corresponding duality gaps at the end of optimization.
        """
        _, coefs, dual_gaps = celer_path(
            X, y, "logreg", alphas=1. / Cs, coef_init=coef_init,
            max_iter=self.max_iter, max_epochs=self.max_epochs,
            p0=self.p0, verbose=self.verbose, tol=self.tol,
            use_PN=(solver == "celer-pn"), X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))
        return coefs, dual_gaps


class GroupLasso(Lasso_sklearn):
    r"""
    Group Lasso scikit-learn estimator based on Celer solver

    The optimization objective for the Group Lasso is::

        (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * \sum_g ||w_g||_2

    where `w_g` is the weight vector of group number `g`.

    Parameters
    ----------
    groups : int | list of ints | list of lists of ints.
        Partition of features used in the penalty on `w`.
        If an int is passed, groups are contiguous blocks of features, of size
        `groups`.
        If a list of ints is passed, groups are assumed to be contiguous,
        group number `g` being of size `groups[g]`.
        If a list of lists of ints is passed, `groups[g]` contains the
        feature indices of the group number `g`.

    alpha : float, optional
        Constant that multiplies the penalty term. Defaults to 1.0.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    max_epochs : int
        Maximum number of BCD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization: the solver runs until the
        duality gap is smaller than ``tol * norm(y) ** 2 / len(y)`` or the
        maximum number of iteration is reached.

    prune : bool, optional (default=True)
        Whether or not to use pruning when growing working sets.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    Examples
    --------
    >>> from celer import GroupLasso
    >>> clf = GroupLasso(alpha=0.5, groups=[[0, 1], [2]])
    >>> clf.fit([[0, 0, 1], [1, -1, 2], [2, 0, -1]], [1, 1, -1])
    GroupLasso(alpha=0.5, fit_intercept=True,
    groups=[[0, 1], [2]], max_epochs=50000, max_iter=100, normalize=False,
    p0=10, prune=True, tol=0.0001, verbose=0, warm_start=False)
    >>> print(clf.coef_)
    [-0.         -0.          0.39285714]
    >>> print(clf.intercept_)
    0.07142857142857145

    See also
    --------
    celer_path
    GroupLassoCV

    References
    ----------
    .. [1] M. Massias, A. Gramfort, J. Salmon
      "Celer: a Fast Solver for the Lasso wit Dual Extrapolation", ICML 2018,
      http://proceedings.mlr.press/v80/massias18a.html

    .. [2] M. Massias, S. Vaiter, A. Gramfort, J. Salmon
      "Dual extrapolation for sparse Generalized Linear Models", JMLR 2020,
      https://arxiv.org/abs/1907.05830
    """

    def __init__(self, groups=1, alpha=1., max_iter=100,
                 max_epochs=50000, p0=10, verbose=0, tol=1e-4, prune=True,
                 fit_intercept=True, normalize=False, warm_start=False):
        super(GroupLasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.groups = groups
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True,
             **kwargs):
        """Compute Group Lasso path with Celer."""
        results = celer_path(
            X, y, "grouplasso", alphas=alphas, groups=self.groups,
            coef_init=coef_init, max_iter=self.max_iter,
            return_n_iter=return_n_iter, max_epochs=self.max_epochs,
            p0=self.p0, verbose=self.verbose, tol=self.tol, prune=self.prune,
            X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))

        return results


class GroupLassoCV(LassoCV, LinearModelCV):
    r"""
    GroupLassoCV scikit-learn estimator based on Celer solver

    The best model is selected by cross-validation.

    The optimization objective for the Group Lasso is::

        (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * \sum_g ||w_g||_2

    where `w_g` is the weight vector of group number `g`.

    Parameters
    ----------
    groups : int | list of ints | list of lists of ints.
        Partition of features used in the penalty on `w`.
        If an int is passed, groups are contiguous blocks of features, of size
        `groups`.
        If a list of ints is passed, groups are assumed to be contiguous,
        group number `g` being of size `groups[g]`.
        If a list of lists of ints is passed, `groups[g]` contains the
        feature indices of the group number `g`.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path.

    alphas : numpy array, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    fit_intercept : boolean, default True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    tol : float, optional
        Stopping criterion for the optimization: the solver runs until the
        duality gap is smaller than ``tol * norm(y) ** 2 / len(y)`` or the
        maximum number of iteration is reached.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, sklearn `KFold` is used.

    verbose : bool or integer
        Amount of verbosity.

    max_epochs : int, optional (default=50000)
        Maximum number of coordinate descent epochs when solving a subproblem.

    p0 : int, optional (default=10)
        Number of features in the first working set.

    prune : bool, optional (default=True)
        Whether to use pruning when growing the working sets.

    precompute : ignored parameter, kept for sklearn compatibility.

    positive : bool, optional (default=False)
        When set to True, forces the coefficients to be positive.

    n_jobs : int or None, optional (default=None)
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    intercept_ : float
        independent term in decision function.

    mse_path_ : array, shape (n_alphas, n_folds)
        mean square error for the test set on each fold, varying alpha

    alphas_ : numpy array, shape (n_alphas,)
        The grid of alphas used for fitting

    dual_gap_ : ndarray, shape ()
        The dual gap at the end of the optimization for the optimal alpha
        (``alpha_``).

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    See also
    --------
    celer_path
    GroupLasso
    """

    def __init__(self, groups=None, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, max_iter=100,
                 tol=1e-4, cv=None, verbose=0, max_epochs=50000, p0=10,
                 prune=True, precompute='auto', positive=False, n_jobs=None):
        super(GroupLassoCV, self).__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas, max_iter=max_iter,
            tol=tol, cv=cv, fit_intercept=fit_intercept, normalize=normalize,
            verbose=verbose, n_jobs=n_jobs)
        self.groups = groups
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.precompute = precompute

    def path(self, X, y, alphas, coef_init=None, **kwargs):
        """Compute GroupLasso path with Celer."""
        alphas, coefs, dual_gaps = celer_path(
            X, y, "grouplasso", alphas=alphas, groups=self.groups,
            coef_init=coef_init, max_iter=self.max_iter,
            max_epochs=self.max_epochs, p0=self.p0, verbose=self.verbose,
            tol=self.tol, prune=self.prune, positive=self.positive,
            X_scale=kwargs.get('X_scale', None),
            X_offset=kwargs.get('X_offset', None))
        return alphas, coefs, dual_gaps

    def _get_estimator(self):
        return GroupLasso()

    def _is_multitask(self):
        return False
