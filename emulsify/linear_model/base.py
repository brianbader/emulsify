"""Base methods for linear models."""

import pandas as pd
from sklearn.utils.validation import check_is_fitted


class BaseLinear:
    def set_engine(self, engine='sklearn', **kwargs):
        self.engine = engine
        # ToDo: add engine validation check
        # ToDo: set up environment (H20, spark, and probably vowpall-wabbit)
        self.model_kwargs = kwargs or {}
        if engine == 'statsmodels':
            self._statsmodels_engine()
        else:
            self._sklearn_engine()

    def _statsmodels_engine(self):
        raise ValueError("Not implemented for statsmodels API")

    def _sklearn_engine(self):
        raise ValueError("Not implemented for scikit-learn API")

    def tidy(self):
        check_is_fitted(self, 'estimates_')
        tidy_df = pd.DataFrame({'term': self.names_, 'estimate': self.estimates_, 'std.error': self.std_errors_,
                                'statistic': self.z_scores_, 'p.value': self.p_values_})
        return tidy_df
