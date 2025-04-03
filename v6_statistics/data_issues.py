import pandas as pd

from typing import Any, List
from vantage6.algorithm.tools.util import warn


def validate_column(func):
    """
    Decorator to validate if a column exists in the data. If the column is not
    available, a warning message is displayed and a None is returned.

    Parameters:
    func (callable): The function to be decorated.

    Returns:
    callable: The wrapped function that includes input validation.

    Example:
        @validate_column
        def compute_local_stat(df, column):
            # Function logic here
            pass
    """
    def wrapper(df: pd.DataFrame, column: str, *args: Any, **kwargs: Any):
        if column not in df.columns:
            warn(f'Column {column} not found in the data!')
            return
        return func(df, column, *args, **kwargs)
    return wrapper


def strip_invalid_results(func):
    """
    Decorator that strips invalid results from the local computations.

    Parameters:
    func (callable): The function to be decorated.

    Returns:
    callable: The wrapped function that includes input validation.

    Example:
        @strip_invalid_results
        def compute_federated_stat(local_results):
            # Function logic here
            pass
    """
    def wrapper(results: List[Any], *args: Any, **kwargs: Any):
        results = [result for result in results if result is not None]
        return func(results, *args, **kwargs)
    return wrapper
