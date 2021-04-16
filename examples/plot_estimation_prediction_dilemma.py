"""
============================================================
Estimation vs prediction dilemma for Lasso and AdaptiveLasso
============================================================

The example highlights the inability of the Lasso to simultaneously estimate
the support and perform correct prediction. On the contrary, the
AdaptiveLasso succeeds.
"""
import numpy as np
import matplotlib.pyplot as plt

from celer import Lasso, AdaptiveLasso
from celer.datasets import make_correlated_data
from celer.plot_utils import configure_plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, f1_score

configure_plt()

n_samples, n_features = 500, 2000
X, y, w_true = make_correlated_data(
    n_samples, n_features, corr=0.3, snr=3, density=0.05, random_state=0)


def scoring(estimator, X_test, y_test):
    return {'f1': f1_score(estimator.coef_ != 0, w_true != 0),
            'mse': mean_squared_error(estimator.predict(X_test), y_test)}


alpha_max = np.max(np.abs(X.T @ y)) / len(y)
alphas = np.geomspace(alpha_max, alpha_max / 300, num=100)


###############################################################################
# Separate computation and plots
models = [Lasso(fit_intercept=False, warm_start=True),
          AdaptiveLasso(fit_intercept=False, warm_start=True)
          ]

cvs = []

for model in models:
    cvs.append(
        GridSearchCV(model, param_grid={'alpha': alphas}, scoring=scoring,
                     n_jobs=-1, cv=5, refit=False).fit(X, y))

###############################################################################
# Estimation and prediction performances of the two estimators : the Lasso's
# optimal lambda is not the same for estimation and prediction. For the
# AdaptiveLasso, they are almost the same.

fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')
for ix, cv in enumerate(cvs):
    axarr = axes[:, ix]

    for fold in range(5):
        axarr[0].semilogx(alphas / alphas[0],
                          cv.cv_results_[f"split{fold}_test_f1"])
        axarr[1].semilogx(alphas / alphas[0],
                          cv.cv_results_[f"split{fold}_test_mse"])

    axarr[0].semilogx(alphas / alphas[0],
                      cv.cv_results_['mean_test_f1'],
                      label='mean across folds', color='k', lw=2)
    axarr[1].semilogx(alphas / alphas[0],
                      cv.cv_results_['mean_test_mse'],
                      label='mean across folds', color='k', lw=2)
    axarr[0].axvline(
        alphas[np.argmax(cv.cv_results_['mean_test_f1'])] / alphas[0],
        linestyle='--', lw=3, color='k', label=r'best $\lambda$')
    axarr[1].axvline(
        alphas[np.argmin(cv.cv_results_['mean_test_mse'])] / alphas[0],
        linestyle='--', lw=3, color='k', label=r'best $\lambda$')

    axarr[0].set_title(f"{models[ix].__class__.__name__}")
    axarr[0].set_ylabel("F1 score on support")
    axarr[1].set_ylabel("MSE")
    axarr[1].set_xlabel(r'$\lambda / \lambda_{\mathrm{max}}$')
    axarr[0].legend()
    axarr[1].legend()

plt.suptitle("5 fold cross validation metrics")
plt.show(block=False)
