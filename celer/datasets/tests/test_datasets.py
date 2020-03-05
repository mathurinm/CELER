import numpy as np

from celer.datasets import load_libsvm, load_climate


def test_news20():
    # other datasets are super big
    # dataset = np.random.choice(list(NAMES.keys()))
    # print(dataset)
    dataset = 'news20'
    X, y = load_libsvm(dataset)
    np.testing.assert_equal(X.shape[0], y.shape[0])


def test_climate():
    X, y = load_climate()
    np.testing.assert_equal(X.shape[0], y.shape[0])
