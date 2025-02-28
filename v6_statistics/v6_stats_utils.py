import json

import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, List, Union, Tuple
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data


def calculate_column_stats(
        client: AlgorithmClient,
        ids: List[int],
        statistics: Dict[str, List[str]]
) -> Dict[str, Dict[str, Union[int, float, Dict[str, Union[int, float]]]]]:
    """Calculate desired statistics per column

    Parameters:
    - client: Vantage6 client object
    - ids: List of organization IDs
    - statistics: Dictionary with columns and statistics to compute per column

    Returns:
    - Dictionary containing desired statistics per column
    """
    # Storing methods in a dictionary to easily call them
    methods = {
        'counts': compute_federated_counts,
        'minmax': compute_federated_minmax,
        'mean': compute_federated_mean,
        'quantiles': compute_federated_quantiles
    }

    # Computing federated statistics per column
    column_stats = {}
    for column, col_stats in statistics.items():
        info(f'Computing statistics for {column}')
        column_stats[column] = {}
        for statistic in col_stats:
            info(f'Computing {statistic} for {column}')
            column_stats[column][statistic] = methods[statistic](
                client=client, ids=ids, column=column
            )

    return column_stats


@data(1)
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


@data(1)
def compute_local_quantile(df: pd.DataFrame, column: str, q: float) -> float:
    """Compute local quantile

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute quantile
    - q: Quantile to compute

    Returns:
    - Local quantile (float)
    """
    info('Computing local quantile')
    return np.quantile(df[column].dropna().values, q)


def compute_federated_quantiles(
        client: AlgorithmClient, ids: List[int], column: str
) -> Dict[str, float]:
    """Compute federated quantiles

    Parameters:
    - client: Vantage6 client object
    - ids: List of organization IDs
    - column: Name of the column to compute federated quantiles

    Returns:
    - Dictionary with federated quantiles and their standard errors
    """
    quantile = {1: 0.25, 2: 0.50, 3: 0.75}
    federated_quantiles = {}
    for i in range(1, 4):
        info(f'Collecting local quantile Q{i}')
        method_kwargs = dict(column=column, q=quantile[i])
        method = 'compute_local_quantile'
        local_quantiles = launch_subtask(client, method, ids, **method_kwargs)
        quantiles_i = np.array(local_quantiles)

        info(f'Collecting local sampling variances of quantile Q{i}')
        # TODO: add number of iterations for bootstrapping as parameter
        method_kwargs = dict(column=column, q=quantile[i])
        method = 'compute_local_quantile_sampling_variance'
        local_quantile_sampling_variances = launch_subtask(
            client, method, ids, **method_kwargs
        )
        variances_i = np.array(local_quantile_sampling_variances)

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


@data(1)
def compute_local_nrows(df: pd.DataFrame, column: str) -> int:
    """Compute local number of rows

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute number of rows

    Returns:
    - Local number of rows (int)
    """
    info('Computing local number of rows')
    return int(df[column].dropna().count())


def compute_federated_mean(
        client: AlgorithmClient, ids: List[int], column: str
) -> float:
    """Compute federated mean

    Parameters:
    - client: Vantage6 client object
    - ids: List of organization IDs
    - column: Name of the column to compute federated mean

    Returns:
    - Federated mean (float)
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

    return federated_mean


@data(1)
def compute_local_counts(df: pd.DataFrame, column: str) -> Dict[str, int]:
    """Compute local counts per category

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute counts per category

    Returns:
    - Dictionary with local counts per category
    """
    info('Computing local counts per category')
    return df[column].value_counts(dropna=False).to_json()


def compute_federated_counts(
        client: AlgorithmClient, ids: List[int], column: str
) -> Dict[str, int]:
    """Compute federated counts for categorical variables

    Parameters:
    - client: Vantage6 client object
    - ids: List of organization IDs
    - column: Name of the column to compute federated counts

    Returns:
    - Dictionary with federated counts of categorical variables
    """
    info('Collecting local counts per category')
    method_kwargs = dict(column=column)
    method = 'compute_local_counts'
    local_counts = launch_subtask(client, method, ids, **method_kwargs)

    info('Computing federated counts')
    federated_counts = {}
    for local_count in local_counts:
        local_count = json.loads(local_count)
        for key, value in local_count.items():
            if key in federated_counts:
                federated_counts[key] += value
            else:
                federated_counts[key] = value

    return federated_counts


@data(1)
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
    info('Computing local minimum and maximum values')
    return df[column].dropna().min(), df[column].dropna().max()


def compute_federated_minmax(
        client: AlgorithmClient, ids: List[int], column: str
) -> Dict[str, Union[int, float]]:
    """Compute federated minimum and maximum values

    Parameters:
    - client: Vantage6 client object
    - ids: List of organization IDs
    - column: Name of the column to compute federated minimum and maximum

    Returns:
    - Dictionary with federated minimum and maximum values
    """
    info('Collecting local minimum and maximum values')
    method_kwargs = dict(column=column)
    method = 'compute_local_minmax'
    local_minmax = launch_subtask(client, method, ids, **method_kwargs)

    info('Computing federated minimum and maximum values')
    federated_min = np.min(local_minmax)
    federated_max = np.max(local_minmax)

    return {
        'min': federated_min,
        'max': federated_max
    }


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
