import json

import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, List, Union, Tuple
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info, warn, error, get_env_var
from vantage6.algorithm.tools.decorators import data


def calculate_column_stats(
        client: AlgorithmClient,
        ids: List[int],
        statistics: Dict[str, List[str]],
        filter_value: str = None
) -> Dict[str, Dict[str, Union[int, Dict[str, Union[int, float]]]]]:
    """Calculate desired statistics per column

    Parameters:
    - client: Vantage6 client object
    - ids: List of organization IDs
    - statistics: Dictionary with columns and statistics to compute per column
    - filter_value: Value to filter on a column set on node configuration

    Returns:
    - Dictionary containing desired statistics per column
    """
    # Computing local statistics
    method_kwargs = dict(statistics=statistics, filter_value=filter_value)
    method = 'compute_local_stats'
    local_stats = launch_subtask(client, method, ids, **method_kwargs)
    info(f'Local stats: {local_stats}')

    # Storing methods in a dictionary to easily call them
    methods = {
        'counts': compute_federated_counts,
        'minmax': compute_federated_minmax,
        'mean': compute_federated_mean,
        'quantiles': compute_federated_quantiles,
        'nrows': compute_federated_nrows
    }

    # Computing federated statistics per column
    column_stats = {}
    for column, col_stats in statistics.items():
        info(f'Computing federated statistics for {column}')
        column_stats[column] = {}
        for statistic in col_stats:
            info(f'Computing federated {statistic} for {column}')
            input = [
                local_stat[column][statistic] for local_stat in local_stats
            ]
            column_stats[column][statistic] = methods[statistic](input)

    return column_stats


@data(1)
def compute_local_stats(
        df: pd.DataFrame, statistics: Dict[str, List[str]],
        filter_value: str = None
) -> Dict[str, Dict[str, Union[int, str]]]:
    """Compute local statistics per column

    Parameters:
    - df: Input DataFrame
    - statistics: Dictionary with columns and statistics to compute per column
    - filter_value: Value to filter on a column set on node configuration

    Returns:
    - Dictionary containing desired statistics per column
    """
    # Filtering data
    if filter_value:
        df = filter_df(df, filter_value)

    # Storing methods in a dictionary to easily call them
    methods = {
        'counts': compute_local_counts,
        'minmax': compute_local_minmax,
        # 'mean': compute_federated_mean,
        'quantiles': compute_local_quantiles,
        'nrows': compute_local_nrows
    }

    # Computing local statistics per column
    local_stats = {}
    for column, col_stats in statistics.items():
        info(f'Computing statistics for {column}')
        local_stats[column] = {}
        for statistic in col_stats:
            info(f'Computing {statistic} for {column}')
            local_stats[column][statistic] = methods[statistic](
                df=df, column=column
            )

    return local_stats


def compute_local_quantile_sampling_variance(
        df: pd.DataFrame, column: str, q: float, iterations: int = 1000
) -> float:
    """Estimate local sampling variance of the quantile

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to estimate quantile sampling variance
    - q: Quantile to estimate local sampling variance
    - iterations: Number of times to sample, default is 1000

    Returns:
    - Quantile sampling variance (float)
    """
    quantiles = []
    data = df[column].dropna().values
    n = len(data)

    info('Bootstrapping the quantile')
    np.random.seed(0)
    for _ in range(iterations):
        # Randomly sample with replacement, which means that a value can be
        # drawn multiple times. We also get a sample with the same size as
        # the original.
        sample = np.random.choice(data, size=n, replace=True)
        quantiles.append(np.quantile(sample, q))

    info('Estimating local sampling variance of the quantile')
    quantile_variance = np.var(quantiles)

    return quantile_variance


def compute_local_quantiles(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Compute local quantiles

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute quantiles

    Returns:
    - Dictionary with local quantiles and their sampling variances
    """
    q = {1: 0.25, 2: 0.50, 3: 0.75}
    quantiles = {}
    for i in range(1, 4):
        info(f'Collecting local quantile Q{i}')
        quantiles[f'Q{i}'] = np.quantile(df[column].dropna().values, q[i])

        info(f'Collecting local sampling variances of quantile Q{i}')
        # TODO: add number of iterations for bootstrapping as parameter
        quantiles[f'variance_Q{i}'] = compute_local_quantile_sampling_variance(
            df, column, q[i]
        )
    return quantiles


def compute_federated_quantiles(
        local_quantiles: List[Dict[str, float]]
) -> Dict[str, float]:
    """Compute federated quantiles

    Parameters:
    - local_quantiles: List with local quantiles and their sampling variances

    Returns:
    - Dictionary with federated quantiles and their standard errors
    """
    federated_quantiles = {}
    for i in range(1, 4):
        info(f'Unwrapping quantiles Q{i} and their sampling variances')
        quantiles_i = np.array([q[f'Q{i}'] for q in local_quantiles])
        variances_i = np.array([q[f'variance_Q{i}'] for q in local_quantiles])

        info('Computing between study heterogeneity')
        # Using DerSimonian and Laird method to estimate tau2, see equation 8 in
        # https://doi.org/10.1016/j.cct.2006.04.004
        k = len(quantiles_i)
        omega_i0 = 1./pow(variances_i, 2)
        quantile_0 = np.sum(omega_i0*quantiles_i)/np.sum(omega_i0)
        tau2_nom = np.sum(omega_i0*pow((quantiles_i - quantile_0), 2)) - (k-1)
        tau2_den = np.sum(omega_i0) - np.sum(pow(omega_i0, 2))/np.sum(omega_i0)
        tau2 = np.max([0, tau2_nom/tau2_den])

        info(f'Computing federated quantile Q{i} and its standard error')
        # Using approach from McGrath et al. (2019), section 2,
        # see: https://doi.org/10.1002/sim.8013
        omega_i = 1./(variances_i + tau2)
        federated_quantile = np.sum(quantiles_i*omega_i)/np.sum(omega_i)
        federated_quantile_std_err = np.sqrt(1./np.sum(omega_i))
        federated_quantiles[f'Q{i}'] = federated_quantile
        federated_quantiles[f'Q{i}_std_err'] = federated_quantile_std_err

    return federated_quantiles


@data(1)
def compute_local_sum(df: pd.DataFrame, column: str) -> Union[float, int]:
    """Compute local sum

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute sum

    Returns:
    - Local sum (float or int)
    """
    info('Computing local sum')
    return df[column].sum()


def compute_local_nrows(
        df: pd.DataFrame, column: str, dropna: bool = False
) -> int:
    """Compute local number of rows

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute number of rows
    - dropna: Whether to drop nan rows, defaults to False

    Returns:
    - Local number of rows (int)
    """
    if dropna:
        nrows = len(df[column].dropna())
    else:
        nrows = len(df[column])
    return int(nrows)


@data(1)
def compute_local_sum_errors2(
        df: pd.DataFrame, column: str, mean: float
) -> Union[float, int]:
    """Compute local sum of squared errors

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute sum of squared errors
    - mean: Mean to compute local sum of squared errors

    Returns:
    - Local sum of squared errors (float or int)
    """
    info('Computing local sum of squared errors')
    return np.sum((df[column].dropna().values - mean)**2)


def compute_federated_mean(
        client: AlgorithmClient, ids: List[int], column: str
) -> Dict[str, float]:
    """Compute federated mean

    Parameters:
    - client: Vantage6 client object
    - ids: List of organization IDs
    - column: Name of the column to compute federated mean

    Returns:
    - Dictionary with federated mean and its standard deviation
    """
    info('Collecting local sum')
    method_kwargs = dict(column=column)
    method = 'compute_local_sum'
    local_sums = launch_subtask(client, method, ids, **method_kwargs)

    info('Collecting local number of rows')
    method_kwargs = dict(column=column)
    method = 'compute_local_nrows'
    local_nrows = launch_subtask(client, method, ids, **method_kwargs)

    info('Computing federated mean')
    federated_mean = np.sum(local_sums)/np.sum(local_nrows)

    info('Collecting local sum of squared errors')
    method_kwargs = dict(column=column, mean=federated_mean)
    method = 'compute_local_sum_errors2'
    local_sum_errors2 = launch_subtask(client, method, ids, **method_kwargs)

    info('Computing federated standard deviation')
    federated_std = np.sqrt(np.sum(local_sum_errors2)/np.sum(local_nrows))

    return {
        'mean': federated_mean,
        'std': federated_std
    }


def compute_federated_nrows(local_nrows: List[int]) -> int:
    """Compute federated number of rows

    Parameters:
    - local_nrows: List of local number of rows

    Returns:
    - Federated number of rows (int)
    """
    return int(np.sum(local_nrows))


def compute_local_counts(df: pd.DataFrame, column: str) -> Dict[str, int]:
    """Compute local counts per category

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute counts per category

    Returns:
    - Dictionary with local counts per category
    """
    return df[column].value_counts(dropna=False).to_json()


def compute_federated_counts(local_counts: List[str]) -> Dict[str, int]:
    """Compute federated counts for categorical variables

    Parameters:
    - local_counts: List of local counts per category

    Returns:
    - Dictionary with federated counts of categorical variables
    """
    federated_counts = {}
    for local_count in local_counts:
        local_count = json.loads(local_count)
        for key, value in local_count.items():
            if key in federated_counts:
                federated_counts[key] += value
            else:
                federated_counts[key] = value
    return federated_counts


def compute_local_minmax(
        df: pd.DataFrame, column: str
) -> Tuple[Union[int, float], Union[int, float]]:
    """Compute local minimum and maximum

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute minimum and maximum

    Returns:
    - Tuple with minimum and maximum values
    """
    return df[column].dropna().min(), df[column].dropna().max()


def compute_federated_minmax(
       local_minmax: List[Tuple[Union[int, float], Union[int, float]]]
) -> Dict[str, Union[int, float]]:
    """Compute federated minimum and maximum values

    Parameters:
    - local_minmax: List of tuples of minimum and maximum values for a column

    Returns:
    - Dictionary with federated minimum and maximum values
    """
    return {
        'min': np.min(local_minmax),
        'max': np.max(local_minmax)
    }


def filter_df(df: pd.DataFrame, filter_value: str) -> pd.DataFrame:
    """ Filter a DataFrame based on a specified value for a column specified via
    node configuration.

    Parameters:
    - df: Input DataFrame
    - filter_value: Value to filter on the DataFrame using the specified column
    - [Env var] V6_FILTER_COLUMN: Name of the column to filter on
    - [Env var] V6_FILTER_VALUES_ALLOWED: Allowed values for the filter column

    Returns:
    - Filtered DataFrame
    """
    filter_column = get_env_var('V6_FILTER_COLUMN', default=False)
    if filter_column is False:
        error(
            'Filtering requested, but no filter column is set in the node configuration.'
            ' Please set the V6_FILTER_COLUMN environment variable at the node.'
        )
        raise ValueError('No filter column set')
    else:
        if filter_column not in df.columns:
            error(
                f'Filter column "{filter_column}" not found in dataset. '
                f'Please check the column name and the dataset.'
            )
            raise ValueError('Filter column not found in dataset')

    filters_allowed = get_env_var('V6_FILTER_VALUES_ALLOWED', default=False)
    if filters_allowed is False:
        warn('No limitations on filter values are set. All values are allowed.')
    else:
        # Parse env var V6_FILTER_VALUES_ALLOWED
        filters_allowed = filters_allowed.split(',')
        if filter_value not in filters_allowed:
            error(
                f'Filter value "{filter_value}" is not allowed. '
                f'Allowed values are: {", ".join(filters_allowed)}'
            )
            raise ValueError('Filter value not allowed')

    # If type of filter_column is not string, convert it to string
    # TODO: this is a temporary solution, we need to handle other types
    if not pd.api.types.is_string_dtype(df[filter_column]):
        df[filter_column] = df[filter_column].astype(str)

    return df[df[filter_column] == str(filter_value)]


def launch_subtask(
    client: AlgorithmClient,
    method: Callable[[Any], Any],
    ids: List[int],
    **kwargs
) -> List[Dict[str, Union[str, List[str]]]]:
    """Launches a subtask to multiple organizations and waits for results

    Parameters:
    - client: The Vantage6 client object used for communication with the server
    - method: The callable to be executed as a subtask by the organizations
    - ids: A list of organization IDs to which the subtask will be distributed
    - **kwargs: Additional keyword arguments to be passed to the method

    Returns:
    - A list of dictionaries containing results obtained from the organizations
    """
    info(f'Sending task to organizations {ids}')
    task = client.task.create(
        input_={
            'method': method,
            'kwargs': kwargs
        },
        organizations=ids
    )
    info('Waiting for results')
    results = client.wait_for_results(task_id=task.get('id'), interval=1)
    info(f'Results obtained for {method}!')
    return results
