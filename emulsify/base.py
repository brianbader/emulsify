"""Base methods for all models."""

from abc import ABC, abstractmethod
from patsy import dmatrices


class BaseMixin(ABC):
    @abstractmethod
    def fit(self, X, y, **fit_params):
        return

    def fit_from_formula(self, formula, data, **fit_params):
        # ToDo: figure out how to handle other input types (numpy?)
        y, X = dmatrices(formula, data=data, return_type="dataframe")
        self.fit(X, y, **fit_params)
