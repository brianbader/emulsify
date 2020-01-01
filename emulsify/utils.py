"""Utility methods and helper fns for all models."""


def _get_column_names(X, fit_intercept):
    col_names = []

    if fit_intercept:
        col_names.append('(Intercept)')

    if hasattr(X, 'columns'):
        col_names.extend(list(X.columns))
    else:
        col_names.extend([f"x{i}" for i in range(X.shape[1])])

    return col_names
