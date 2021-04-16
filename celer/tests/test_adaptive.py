# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import numpy as np
from numpy.linalg import norm
# from sklearn.utils.estimator_checks import check_estimator

from celer import Lasso, LassoCV, AdaptiveLasso, AdaptiveLassoCV
from celer.utils.testing import build_dataset

from scipy.linalg import toeplitz
from sklearn.utils import check_random_state

# def test_adaptive_lasso_class():
# check_estimator(AdaptiveLasso())
# check_estimator(AdaptiveLassoCV())


def test_adaptive_lasso():
    sparse_X = False
    X, y = build_dataset(
        n_samples=60, n_features=100, sparse_X=sparse_X)
    alpha_max = norm(X.T.dot(y), ord=np.inf) / len(y)
    alpha = alpha_max / 10
    tol = 1e-10
    n_reweightings = 5
    adalasso = AdaptiveLasso(alpha=alpha, n_reweightings=n_reweightings,
                             fit_intercept=False, tol=tol)
    adalasso.fit(X, y)

    lasso = Lasso(alpha=alpha, warm_start=False,
                  weights=None, fit_intercept=False, tol=tol)
    reweights = np.ones(X.shape[1])
    for _ in range(n_reweightings):
        lasso.weights = reweights
        lasso.fit(X, y)
        reweights = 1. / np.abs(lasso.coef_)

    np.testing.assert_allclose(lasso.coef_, adalasso.coef_)


def breaking_test():
    n_samples, n_features = 5, 10
    rng = check_random_state(0)
    X = rng.multivariate_normal(size=n_samples, mean=np.zeros(n_features),
                                cov=toeplitz(0.7 ** np.arange(n_features)))

    w_true = np.zeros(n_features)
    size_supp = 20
    w_true[::n_features // size_supp] = (-1) ** np.arange(size_supp)
    noise = rng.randn(n_samples)
    y = X @ w_true
    y += noise / norm(noise) * 0.5 * norm(y)

    AdaptiveLassoCV(n_jobs=1, verbose=1).fit(X, y)


if __name__ == "__main__":
    n_samples, n_features = 10, 10
    rng = check_random_state(0)
    X = rng.multivariate_normal(size=n_samples, mean=np.zeros(n_features),
                                cov=toeplitz(0.7 ** np.arange(n_features)))

    X = np.asfortranarray(X)
    w_true = np.zeros(n_features)
    size_supp = min(20, n_features)
    w_true[:size_supp] = (-1) ** np.arange(size_supp)
    noise = rng.randn(n_samples)
    y = X @ w_true
    y += noise / norm(noise) * 0.5 * norm(y)

    lasso = LassoCV(fit_intercept=False, n_jobs=-1).fit(X, y)
    alphas = lasso.alphas_

    1 / 0
    # res = celer_path(X, y, "lasso", n_reweightings=5, verbose=1)

    AdaptiveLassoCV(n_jobs=1, verbose=0, n_reweightings=2).fit(X, y)
