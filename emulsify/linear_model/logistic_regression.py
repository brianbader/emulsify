"""Wrappers for logistic regression implementations."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression as ScikitLR
from statsmodels.api import Logit as StatsModelsLR  # change to MNLogit
import scipy.stats as stat
import statsmodels.api as sm

from emulsify.utils import _get_column_names
from emulsify.linear_model.base import BaseLinear


# change to base method later
class LogisticRegression(BaseEstimator, BaseLinear):
    def __init__(self, penalty='l2', C=1.0, fit_intercept=True, tol=0.0001, max_iter=100, engine='scikit-learn',
                 **kwargs):
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.kwargs = kwargs or {}
        self.engine = engine
        # ToDo: set up environment (H20, spark, and probably vowpall-wabbit)

    def _initialize(self, X, y):
        # ToDo: add engine validation check
        if self.engine == 'statsmodels':
            X_ = sm.add_constant(X) if self.fit_intercept else X.copy()
            self.model = StatsModelsLR(exog=X_, endog=y, **self.kwargs)
        else:
            self.model = ScikitLR(penalty=self.penalty, C=self.C, fit_intercept=self.fit_intercept,
                                  tol=self.tol, max_iter=self.max_iter, **self.kwargs)

    def _fit_case(self):
        fit_case = {
            'scikit-learn': self._fit_sklearn,
            'statsmodels': self._fit_statsmodels,
        }
        return fit_case

    def fit(self, X, y, sample_weight=None, **fit_params):
        self._initialize(X, y)
        self.names_ = _get_column_names(X, self.fit_intercept)
        self._fit_case()[self.engine](X, y, sample_weight, **fit_params)
        self.z_scores_ = self.estimates_ / self.std_errors_
        self.p_values_ = np.array([stat.norm.sf(abs(z)) * 2 for z in self.z_scores_])
        return self

    def _fit_sklearn(self, X, y, sample_weight, **fit_params):
        self.model.fit(X, y, sample_weight)

        self.estimates_ = self.model.coef_.ravel()
        if self.fit_intercept:
            self.estimates_ = np.concatenate((self.model.intercept_, self.estimates_))

        # borrowed from this: gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        X_ = np.hstack([np.ones((X.shape[0], 1)), X]) if self.fit_intercept else X.copy()
        denom = np.tile(denom, (X_.shape[1], 1)).T

        f_ij = np.dot((X_ / denom).T, X_)  # fisher information matrix
        cramer_rao = np.linalg.inv(f_ij)   # inverse fisher matrix

        self.std_errors_ = np.sqrt(np.diagonal(cramer_rao))
        self.n_iter_ = self.model.n_iter_

    def _fit_statsmodels(self, X, y, sample_weight, **fit_params):
        if self.penalty == 'l1':
            logit_fitted = self.model.fit_regularized(disp=False, maxiter=self.max_iter, alpha=1/self.C, **fit_params)
        else:
            if self.penalty != 'none':
                Warning('Fitting unregularized model...')
            logit_fitted = self.model.fit(disp=False, maxiter=self.max_iter, **fit_params)

        self.estimates_ = logit_fitted.params
        self.n_iter_ = np.asarray(logit_fitted.mle_retvals['iterations'])
        self.std_errors_ = np.sqrt(np.diag(logit_fitted.normalized_cov_params))