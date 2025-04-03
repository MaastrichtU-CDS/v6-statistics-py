import json

import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, List, Union, Tuple
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info, warn, error, get_env_var
from vantage6.algorithm.tools.decorators import data
from .data_issues import validate_column, strip_invalid_results


def calculate_column_stats(
        client: AlgorithmClient,
        ids: List[int],
        statistics: Dict[str, List[str]],
        suppression: int = None,
        filter_value: str = None
) -> Dict[str, Dict[str, Union[int, Dict[str, Union[int, float]]]]]:
    """Calculate desired statistics per column

    Parameters:
    - client: Vantage6 client object
    - ids: List of organization IDs
    - statistics: Dictionary with columns and statistics to compute per column
    - suppression: Number of records to apply suppression
    - filter_value: Value to filter on a column set on node configuration

    Returns:
    - Dictionary containing desired statistics per column
    """
    # Computing local statistics
    method_kwargs = dict(statistics=statistics, filter_value=filter_value)
    method = 'compute_local_stats'
    local_stats = launch_subtask(client, method, ids, **method_kwargs)

    # Storing methods in a dictionary to easily call them
    methods = {
        'counts': compute_federated_counts,
        'minmax': compute_federated_minmax,
        'mean': compute_federated_mean,
        'quantiles': compute_federated_quantiles,
        'nrows': compute_federated_nrows,
        'nans': compute_federated_nans
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
            column_stats[column][statistic] = methods[statistic](
                input, suppression
            )

    # Computing local standard deviations
    method_kwargs = dict(
        column_stats=column_stats,
        filter_value=filter_value
    )
    method = 'compute_local_stds'
    local_stds = launch_subtask(client, method, ids, **method_kwargs)

    # Computing federated standard deviations
    for column, col_stats in statistics.items():
        if 'mean' in col_stats:
            info(f'Computing federated standard deviation for {column}')
            stds = [local_std[column] for local_std in local_stds]
            column_stats[column]['mean']['std'] = compute_federated_std(stds)

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
        'mean': compute_local_means,
        'quantiles': compute_local_quantiles,
        'nrows': compute_local_nrows,
        'nans': compute_local_nans
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


@validate_column
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


@validate_column
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
        values = df[column].dropna().values
        quantiles[f'Q{i}'] = np.quantile(values, q[i]) \
            if len(values) > 0 else np.nan

        info(f'Collecting local sampling variances of quantile Q{i}')
        # TODO: add number of iterations for bootstrapping as parameter
        quantiles[f'variance_Q{i}'] = compute_local_quantile_sampling_variance(
            df, column, q[i]
        ) if len(values) > 0 else np.nan
    quantiles['nrows'] = compute_local_nrows(df, column, True)
    return quantiles


@strip_invalid_results
def compute_federated_quantiles(
        local_quantiles: List[Dict[str, float]], suppression: int = None
) -> Dict[str, float]:
    """Compute federated quantiles

    Parameters:
    - local_quantiles: List with local quantiles and their sampling variances
    - suppression: Number of records to apply suppression

    Returns:
    - Dictionary with federated quantiles and their standard errors
    """
    # Applying global suppression
    if suppression:
        nrows = np.sum([
            quantile['nrows'] for quantile in local_quantiles
            if not np.isnan(quantile['nrows'])
        ])
        if nrows <= suppression:
            return {'Q': np.nan}

    # Computing federated quantiles
    federated_quantiles = {}
    for i in range(1, 4):
        info(f'Unwrapping quantiles Q{i} and their sampling variances')
        # Also removing NaN results to account for nodes that have columns
        # with only NaN values
        quantiles_i = np.array([
            q[f'Q{i}'] for q in local_quantiles if not np.isnan(q[f'Q{i}'])
        ])
        variances_i = np.array([
            q[f'variance_Q{i}'] for q in local_quantiles
            if not np.isnan(q[f'variance_Q{i}'])
        ])
        if len(quantiles_i) != len(variances_i):
            error(
                'Length of lists of local quantiles and variances do not match!'
            )

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


@validate_column
def compute_local_sum(df: pd.DataFrame, column: str) -> Union[float, int]:
    """Compute local sum

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute sum

    Returns:
    - Local sum (float or int)
    """
    return float(df[column].sum())


@validate_column
def compute_local_nans(df: pd.DataFrame, column: str) -> int:
    """Compute local number of NaNs in a column

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute number of rows

    Returns:
    - Local number of NaNs (int)
    """
    return int(df[column].isna().sum())


@validate_column
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


@validate_column
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
    return np.sum((df[column].dropna().values - mean)**2)


@validate_column
def compute_local_means(
        df: pd.DataFrame, column: str
) -> Dict[str, Union[float, int]]:
    """Compute local sum and number of non-NA rows for a certain column

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute sum and nrows

    Returns:
    - Dictionary of local sum and number of non-NA rows for a certain column
    """
    return {
        'sum': compute_local_sum(df, column),
        'nrows': compute_local_nrows(df, column, True)
    }

@data(1)
def compute_local_stds(
        df: pd.DataFrame,
        column_stats: Dict[str, Union[int, Dict[str, Union[int, float]]]],
        filter_value: str = None,
) -> Dict[str, Union[float, int]]:
    """Compute local sum of squared errors and number of nonNA rows for a column

    Parameters:
    - df: Input DataFrame
    - column_stats: Dictionary with federated column statistics
    - filter_value: Value to filter on a column set on node configuration

    Returns:
    - Dictionary of sum of squared errors and number of non-NA rows
    """
    # Filtering data
    if filter_value:
        df = filter_df(df, filter_value)

    # Computing local standard deviations per column
    local_stds = {}
    for column, statistics in column_stats.items():
        if 'mean' in statistics.keys():
            info(f'Computing standard deviation for {column}')
            mean = statistics['mean']['mean']
            local_stds[column] = {
                'sum_errors2': compute_local_sum_errors2(df, column, mean),
                'nrows': compute_local_nrows(df, column, True)
            }

    return local_stds


@strip_invalid_results
def compute_federated_mean(
        local_means: List[Dict[str, float]], suppression: int = None
) -> Dict[str, float]:
    """Compute federated mean

    Parameters:
    - local_means: List with local sums and nrows per column
    - suppression: Number of records to apply suppression

    Returns:
    - Federated mean (float)
    """
    # Unwrap local sums and number of rows and remove NaN results to account
    # for nodes that have columns with only NaNs
    local_sums = [
        local_mean['sum'] for local_mean in local_means
        if not np.isnan(local_mean['sum'])
    ]
    local_nrows = [
        local_mean['nrows'] for local_mean in local_means
        if not np.isnan(local_mean['nrows'])
    ]
    if len(local_sums) != len(local_nrows):
        error('Length of lists of local sums and number of rows do not match!')

    # Compute federated mean and apply suppression if necessary and required
    nrows = np.sum(local_nrows)
    if nrows > 0:
        federated_mean = np.sum(local_sums)/nrows
        if suppression:
            if nrows <= suppression:
                federated_mean = np.nan
    else:
        federated_mean = np.nan

    return {
        'mean': federated_mean
    }


def compute_federated_std(local_stds: List[Dict[str, float]]) -> float:
    """Compute federated standard deviation, when suppression is applied for
    the mean it will automatically be applied here

    Parameters:
    - local_stds: List with local sums of squared errors and nrows per column

    Returns:
    - Federated standard deviation (float)
    """
    local_sum_errors2 = [
        local_std['sum_errors2'] for local_std in local_stds
        if local_std['sum_errors2'] is not None
    ]
    local_nrows = [
        local_std['nrows'] for local_std in local_stds
        if local_std['nrows'] is not None
    ]
    # TODO: the result does not exactly match with centralised, why? Does it
    #  have to do with the size of the floats?
    federated_std = np.sqrt(np.sum(local_sum_errors2)/np.sum(local_nrows))
    return federated_std


@strip_invalid_results
def compute_federated_nrows(
        local_nrows: List[int], suppression: int = None
) -> int:
    """Compute federated number of rows

    Parameters:
    - local_nrows: List of local number of rows
    - suppression: Number of records to apply suppression

    Returns:
    - Federated number of rows (int)
    """
    nrows = int(np.sum(local_nrows))
    if suppression:
        if nrows < suppression:
            nrows = suppression
    return nrows


@strip_invalid_results
def compute_federated_nans(
        local_nans: List[int], suppression: int = None
) -> int:
    """Compute federated number of NaNs for a certain variable

    Parameters:
    - local_nans: List of local number of NaNs
    - suppression: Number of records to apply suppression

    Returns:
    - Federated number of NaNs (int)
    """
    nans = int(np.sum(local_nans))
    if suppression:
        if nans < suppression:
            nans = suppression
    return nans


@validate_column
def compute_local_counts(df: pd.DataFrame, column: str) -> Dict[str, int]:
    """Compute local counts per category

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute counts per category

    Returns:
    - Dictionary with local counts per category
    """
    return df[column].value_counts(dropna=False).to_json()


@strip_invalid_results
def compute_federated_counts(
        local_counts: List[str], suppression
) -> Dict[str, int]:
    """Compute federated counts for categorical variables

    Parameters:
    - local_counts: List of local counts per category
    - suppression: Number of records to apply suppression

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

    # Apply suppression in a global level
    if suppression:
        for key, value in federated_counts.items():
            if value < suppression:
                federated_counts[key] = suppression

    return federated_counts


@validate_column
def compute_local_minmax(
        df: pd.DataFrame, column: str
) -> Dict[str, Union[Tuple[Union[int, float], Union[int, float]], int]]:
    """Compute local minimum and maximum

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute minimum and maximum

    Returns:
    - Dictionary with minimum and maximum and number of rows for suppression
    """
    min = df[column].dropna().min()
    max = df[column].dropna().max()

    # We want to make sure we are returning something simple for privacy's sake
    assert isinstance(min, (int, float, np.nan))
    assert isinstance(max, (int, float, np.nan))

    return {
        'minmax': [min, max],
        'nrows': compute_local_nrows(df, column, True)
    }


@strip_invalid_results
def compute_federated_minmax(
        local_minmax: List[Dict[
            str, Union[Tuple[Union[int, float], Union[int, float]], int]
        ]],
        suppression: int = None
) -> Dict[str, Union[int, float]]:
    """Compute federated minimum and maximum values

    Parameters:
    - local_minmax: List of tuples of minimum and maximum values for a column
    - suppression: Number of records to apply suppression

    Returns:
    - Dictionary with federated minimum and maximum values
    """
    # Applying global suppression
    if suppression:
        nrows = np.sum([nrow['nrows'] for nrow in local_minmax])
        if nrows <= suppression:
            return {'min': np.nan, 'max': np.nan}

    # Computing global minimum
    min = [
        min['minmax'][0] for min in local_minmax
        if not np.isnan(min['minmax'][0])
    ]
    min = np.min(min) if len(min) > 0 else np.nan

    # Computing global maximum
    max = [
        max['minmax'][1] for max in local_minmax
        if not np.isnan(max['minmax'][1])
    ]
    max = np.max(max) if len(max) > 0 else np.nan

    return {
        'min': min,
        'max': max
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
