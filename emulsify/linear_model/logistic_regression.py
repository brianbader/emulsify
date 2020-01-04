"""Wrappers for logistic regression implementations."""

import warnings

import numpy as np
import scipy.stats as stat
import statsmodels.api as sm
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression as ScikitLR
from statsmodels.api import MNLogit as StatsModelsLR

from emulsify.utils import _get_column_names
from emulsify.linear_model.base import BaseLinear


# change to base method later
class LogisticRegression(BaseEstimator, BaseLinear):
    def __init__(self, penalty=0., mixture=0., fit_intercept=True, mode='classification'):
        self.penalty = penalty
        self.mixture = mixture
        self.mode = mode
        self.fit_intercept = fit_intercept

    def set_engine(self, engine='sklearn', **kwargs):
        self.engine = engine
        # ToDo: add engine validation check
        # ToDo: set up environment (H20, spark, and probably vowpall-wabbit)
        self.model_kwargs = kwargs or {}
        if engine == 'statsmodels':
            self._statsmodels_engine()
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
        self.model_kwargs['fit_intercept'] = self.fit_intercept
        self.model = ScikitLR(**self.model_kwargs)

    def _statsmodels_engine(self):
        # statsmodels api requires X, y in init step so doesn't initialize until fit is called
        if self.penalty != 0.:
            self.model_kwargs['alpha'] = self.penalty

    def _fit_case(self):
        fit_case = {
            'sklearn': self._fit_sklearn,
            'statsmodels': self._fit_statsmodels,
        }
        return fit_case

    def fit(self, X, y, sample_weight=None, **fit_params):
        self.names_ = _get_column_names(X, self.fit_intercept)
        self._fit_case()[self.engine](X, y, sample_weight, **fit_params)
        self._get_std_errors(X, y)
        self.z_scores_ = self.estimates_ / self.std_errors_
        self.p_values_ = np.array([stat.norm.sf(abs(z)) * 2 for z in self.z_scores_])
        return self

    def _fit_sklearn(self, X, y, sample_weight, **fit_params):
        self.model.fit(X, y, sample_weight)

        self.estimates_ = self.model.coef_.flatten()
        if self.fit_intercept:
            self.estimates_ = np.concatenate((self.model.intercept_, self.estimates_))

        self.num_iterations = self.model.n_iter_

    def _fit_statsmodels(self, X, y, sample_weight=None, **fit_params):
        X_ = sm.add_constant(X) if self.fit_intercept else X.copy()
        self.model = StatsModelsLR(exog=X_, endog=y)

        if self.penalty == 0.:
            logit_fitted = self.model.fit(**self.model_kwargs)
        else:
            warnings.warn('Fitting l1 regularized model with specified penalty... mixture term for statsmodels has no effect')
            logit_fitted = self.model.fit_regularized(**self.model_kwargs)

        self.estimates_ = np.asarray(logit_fitted.params).flatten()
        self.num_iterations = np.asarray(logit_fitted.mle_retvals['iterations'])

    def _get_std_errors(self, X, y):
        # Multinomial logit Hessian matrix of the log-likelihood, from StatsModels API
        self.hessian_ = sm.MNLogit(endog=y, exog=X).hessian(params=self.estimates_)
        self.std_errors_ = np.sqrt(np.diagonal(np.linalg.inv(-self.hessian_)))
