"""Wrappers for logistic regression implementations."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression as ScikitLR
from statsmodels.api import Logit as StatsModelsLR  # change to MNLogit
import scipy.stats as stat
import statsmodels.api as sm


# change to base method later
class LogisticRegression(BaseEstimator):
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

    def _get_estimate_names(self):
        pass

    def fit(self, X, y, sample_weight=None, **fit_params):
        self._initialize(X, y)
        # ToDo: set term names
        self.names = 'test'

        if self.engine == 'statsmodels':
            self._fit_statsmodels(X, y, **fit_params)
        else:
            self._fit_sklearn(X, y, sample_weight, **fit_params)
        return self

    def _fit_sklearn(self, X, y, sample_weight, **fit_params):
        self.model.fit(X, y, sample_weight)

        self.estimates = self.model.coef_.ravel()
        if self.fit_intercept:
            self.estimates = np.concatenate((self.model.intercept_, self.estimates))

        # borrowed from this: gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        X_ = np.hstack([np.ones((X.shape[0], 1)), X]) if self.fit_intercept else X.copy()
        denom = np.tile(denom, (X_.shape[1], 1)).T

        f_ij = np.dot((X_ / denom).T, X_)  # fisher information matrix
        cramer_rao = np.linalg.inv(f_ij)   # inverse fisher matrix

        self.std_errors = np.sqrt(np.diagonal(cramer_rao))
        self.z_scores = self.estimates / self.std_errors
        self.p_values = np.array([stat.norm.sf(abs(z)) * 2 for z in self.z_scores])

    # def _fit_statsmodels(self, X, y, **fit_params):
    #     logit_init = Logit(endog=y, exog=X)
    #     logit_fitted = logit_init.fit(disp=False, **fit_params)
    #     if self.fit_intercept:
    #         self.coef_ = logit_fitted.params[1:]
    #         self.intercept_ = logit_fitted.params[1]
    #     else:
    #         self.coef_ = logit_fitted.params
    #         self.intercept_ = 0.
    #
    #     self.coef_ = np.asarray(self.coef_).reshape(1, len(self.coef_))
    #     self.intercept_ = np.asarray(self.intercept_).reshape(1, )
    #     self.n_iter_ = np.asarray(logit_fitted.mle_retvals['iterations'])
    #     # put p value, z-stat, and std errors
    #
    def tidy(self):
        # ToDo: check is fitted
        tidy_df = pd.DataFrame({'term': self.names, 'estimate': self.estimates, 'std.error': self.std_errors,
                                'statistic': self.z_scores, 'p.value': self.p_values})
        return tidy_df
