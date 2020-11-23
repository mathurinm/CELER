# Authors:
# Mathurin Massias
# Thomas Moreau

import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state


def make_correlated_data(n_samples=100, n_features=50, rho=0.6, snr=3,
                         density=0.2, w_true=None, random_state=None):
    r"""Generate correlated design matrix with decaying correlation rho**|i-j|.
    according to

    .. math::

        y = X w^* + \epsilon

    such that :math:`||X w^*|| / ||\epsilon|| = snr`.

    The generated features have mean 0, variance 1 and the expected correlation
    structure:

    .. math::

        \mathbb E[x_i] = 0~, \quad \mathbb E[x_i^2] = 1  \quad
        \text{and} \quad \mathbb E[x_ix_j] = \rho^{|i-j|}


    Parameters
    ----------
    n_samples: int
        Number of samples in the design matrix.
    n_features: int
        Number of features in the design matrix.
    corr: float
        Correlation :math:`\rho` between successive features. The element
        :math:`C_{i, j}` in the correlation matrix will be
        :math:`\rho^{|i-j|}`. This parameter should be selected in
        :math:`[0, 1[`.
    snr : float
        Signal-to-noise ratio.
    w_true: np.array, shape (n_features,) | None
        True regression coefficients. If None, an array with `nnz` non zero
        standard Gaussian entries is simulated.
    nnz: int
        Number of non zero elements in w_true if it must be simulated.
    random_state : int | RandomState instance | None (default)
        Determines random number generation for data generation. Use an int to
        make the randomness deterministic.

    Returns
    -------
    X: ndarray, shape (n_samples, n_features)
        A design matrix with Toeplitz covariance.
    y: ndarray, shape (n_samples,)
        Observation vector.
    w_true: ndarray, shape (n_features,)
        True regression vector of the model.
    """
    if not 0 <= rho < 1:
        raise ValueError("The correlation `rho` should be chosen in [0, 1[.")
    if not 0 < density <= 1:
        raise ValueError("The density should be chosen in ]0, 1].")
    rng = check_random_state(random_state)
    nnz = int(density * n_features)

    if rho == 0:
        X = np.asfortranarray(rng.randn(n_samples, n_features))
    else:
        # X is generated cleverly using an AR model with reason corr and
        # innovation sigma^2 = 1 - \rho ** 2: X[:, j+1] = rho X[:, j] + eps_j
        # where eps_j = sigma * np.random.randn(n_samples)
        sigma = np.sqrt(1 - rho * rho)
        U = rng.randn(n_samples)

        X = np.empty([n_samples, n_features], order='F')
        X[:, 0] = U
        for j in range(1, n_features):
            U *= rho
            U += sigma * rng.randn(n_samples)
            X[:, j] = U

    if w_true is None:
        w_true = np.zeros(n_features)
        support = rng.choice(n_features, nnz, replace=False)
        w_true[support] = rng.randn(nnz)

    y = X @ w_true
    if snr != 0:
        noise = rng.randn(n_samples)
        y += noise / norm(noise) * norm(y) / snr
    return X, y, w_true
