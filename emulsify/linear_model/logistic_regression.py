"""Wrappers for logistic regression implementations."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from statsmodels.api import Logit  # change to MNLogit
import scipy.stats as stat


# change to base method later
class LogisticRegression(SkLogisticRegression):
    def __init__(self, engine='scikit-learn'):
        self.engine = engine
        # ToDo: add engine validation check
        # ToDo: set up environment (H20, spark,  and probably vowpall-wabbit)
        super().__init__()

    def fit(self, X, y, sample_weight=None, **fit_params):
        if self.engine == 'statsmodels':
            self._fit_statsmodels(X, y, **fit_params)
        else:
            self._fit_sklearn(X, y, sample_weight, **fit_params)
        return self

    def _fit_sklearn(self, X, y, sample_weight, **fit_params):
        super().fit(X, y, sample_weight)

        # borrowed from this: gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d
        denom = (2.0 * (1.0 + np.cosh(self.decision_function(X))))

        # if self.fit_intercept:
        #    X_ = np.hstack([np.ones((X.shape[0], 1)), X])

        denom = np.tile(denom, (X.shape[1], 1)).T

        f_ij = np.dot((X / denom).T, X)  ## Fisher Information Matrix
        cramer_rao = np.linalg.inv(f_ij)  ## Inverse Information Matrix

        # if self._fit_intercept:
        #     self.coef = np.column_stack((self.model.intercept_, self.model.coef_))
        # else:
        #     self.coef = self.model.coef_

        self.std_errs = np.sqrt(np.diagonal(cramer_rao))
        self.z_scores = (self.coef_ / self.sigma)[0]
        self.p_values = [stat.norm.sf(abs(x)) * 2 for x in self.z]

    def _fit_statsmodels(self, X, y, **fit_params):
        logit_init = Logit(endog=y, exog=X)
        logit_fitted = logit_init.fit(disp=False, **fit_params)
        if self.fit_intercept:
            self.coef_ = logit_fitted.params[1:]
            self.intercept_ = logit_fitted.params[1]
        else:
            self.coef_ = logit_fitted.params
            self.intercept_ = 0.

        self.coef_ = np.asarray(self.coef_).reshape(1, len(self.coef_))
        self.intercept_ = np.asarray(self.intercept_).reshape(1, )
        self.n_iter_ = np.asarray(logit_fitted.mle_retvals['iterations'])
        # put p value, z-stat, and std errors

    def tidy(self):
        # ToDo: check is fitted
        pd.DataFrame({'term': self.coef_names, 'estimate': self.coef_})

