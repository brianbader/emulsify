"""Base methods for all models."""

from patsy import dmatrices


class BaseMixin:
    def fit(self, X, y, **fit_params):
        pass

    def fit_from_formula(self, formula, data, **fit_params):
        # ToDo: figure out how to handle other input types (numpy?)
        y, X = dmatrices(formula, data=data, return_type="dataframe")
        self.fit(X, y, **fit_params)

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
