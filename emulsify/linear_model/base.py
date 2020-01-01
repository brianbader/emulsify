"""Base methods for linear models."""

import pandas as pd
from sklearn.utils.validation import check_is_fitted


class BaseLinear:
    def tidy(self):
        check_is_fitted(self, 'estimates_')
        tidy_df = pd.DataFrame({'term': self.names_, 'estimate': self.estimates_, 'std.error': self.std_errors_,
                                'statistic': self.z_scores_, 'p.value': self.p_values_})
        return tidy_df
