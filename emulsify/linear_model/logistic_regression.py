"""Wrappers for logistic regression implementations."""

import numpy as np
import scipy.stats as stat
import statsmodels.api as sm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression as ScikitLR
from statsmodels.api import Logit as StatsModelsLR  # change to MNLogit

from emulsify.utils import _get_column_names
from emulsify.linear_model.base import BaseLinear


# change to base method later
class LogisticRegression(BaseEstimator, BaseLinear):
    def __init__(self, penalty=0., mixture=0., fit_intercept=True, mode='classification'):
        self.penalty = penalty
        self.mixture = mixture
        self.mode = mode
        self.fit_intercept = fit_intercept

    def set_engine(self, engine='scikit-learn', **kwargs):
        self.engine = engine
        # ToDo: add engine validation check
        # ToDo: set up environment (H20, spark, and probably vowpall-wabbit)
        self.model_kwargs = kwargs or {}
        if engine == 'statsmodels':
            self.model = None  # statsmodels api requires X, y in init step
        else:
            self._sklearn_engine()

    def _sklearn_engine(self):
        if self.penalty == 0.:
            self.model_kwargs['penalty'] = 'none'
        elif self.mixture == 1.:
            self.model_kwargs['penalty'] = 'l1'
        elif self.mixture == 0.:
            self.model_kwargs['penalty'] = 'l2'
        else:
            self.model_kwargs['penalty'] = 'elasticnet'
            self.model_kwargs['l1_ratio'] = self.mixture
        if self.penalty != 0.:
            self.model_kwargs['C'] = 1/self.penalty
        self.model = ScikitLR(**self.model_kwargs)

    def _fit_case(self):
        fit_case = {
            'scikit-learn': self._fit_sklearn,
            'statsmodels': self._fit_statsmodels,
        }
        return fit_case

    def fit(self, X, y, sample_weight=None, **fit_params):
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
        X_ = sm.add_constant(X) if self.fit_intercept else X.copy()
        self.model = StatsModelsLR(exog=X_, endog=y, **self.kwargs)

        if self.penalty == 'l1':
            logit_fitted = self.model.fit_regularized(disp=False, maxiter=self.max_iter, alpha=1/self.C, **fit_params)
        else:
            if self.penalty != 'none':
                Warning('Fitting unregularized model...')
            logit_fitted = self.model.fit(disp=False, maxiter=self.max_iter, **fit_params)

        self.estimates_ = logit_fitted.params
        self.n_iter_ = np.asarray(logit_fitted.mle_retvals['iterations'])
        self.std_errors_ = np.sqrt(np.diag(logit_fitted.normalized_cov_params))
