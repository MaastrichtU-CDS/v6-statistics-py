import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, List, Tuple, Union
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import get_env_var, info, warn, error
from vantage6.algorithm.tools.decorators import data


def calculate_column_stats(
        client: AlgorithmClient,
        ids: List[int],
        statistics: Dict[str, List[str]]
) -> Dict[str, Union[str, List[str]]]:
    """Calculate desired statistics per column.

    Parameters:
    - client: Vantage6 client object
    - ids: List of organization IDs
    - statistics: Dictionary with columns and statistics to compute per column

    Returns:
    - Dictionary containing desired statistics per column
    """
    column_stats = {}
    for column, col_stats in statistics.items():
        info(f'Computing statistics for {column}')
        method_kwargs = dict(column=column)
        # Loop through desired statistics per column
        for statistic in col_stats:
            info(f'Computing {statistic} for {column}')
            method = f'compute_federated_{statistic}'
            column_stats[column] = {
                statistic: launch_subtask(client, method, ids, **method_kwargs)
            }
    return column_stats


@data(1)
def compute_local_median_sampling_variance(
        df: pd.DataFrame, column: str, iterations: int = 1000
) -> float:
    """Estimate local sampling variance of the median

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to estimate median sampling variance
    - iterations: Number of times to sample, default is 1000

    Returns:
    - Median sampling variance (float)
    """
    medians = []
    data = df[column].dropna().values
    n = len(data)

    info('Bootstrapping the median')
    for _ in range(iterations):
        # Randomly sample with replacement, which means that a value can be
        # drawn multiple times. We also get a sample with the same size as
        # the original.
        sample = np.random.choice(data, size=n, replace=True)
        medians.append(np.median(sample))

    info('Estimating local sampling variance of the median')
    median_variance = np.var(medians)

    return median_variance


@data(1)
def compute_local_median(df: pd.DataFrame, column: str) -> float:
    """Compute local median

    Parameters:
    - df: Input DataFrame
    - column: Name of the column to compute median

    Returns:
    - Local median (float)
    """
    info('Computing local median')
    return df[column].median()


def compute_federated_median(
        client: AlgorithmClient, ids: List[int], column: str
) -> Dict[str, float]:
    """Compute federated median

    Parameters:
    - client: Vantage6 client object
    - ids: List of organization IDs
    - column: Name of the column to compute federated median

    Returns:
    - Dictionary with federated median and its standard error
    """
    info('Collecting local medians')
    method_kwargs = dict(column=column)
    method = 'compute_local_median'
    local_medians = launch_subtask(client, method, ids, **method_kwargs)
    medians_i = np.array(local_medians)

    info('Collecting local sampling variances of median')
    method_kwargs = dict(column=column)
    method = 'compute_local_median_sampling_variance'
    local_median_sampling_variances = launch_subtask(
        client, method, ids, **method_kwargs
    )
    variances_i = np.array(local_median_sampling_variances)

    info('Computing between study heterogeneity')
    # Using DerSimonian and Laird method to estimate tau2, see equation 8 in
    # https://doi.org/10.1016/j.cct.2006.04.004
    k = len(medians_i)
    omega_i0 = 1./pow(variances_i, 2)
    median_0 = np.sum(omega_i0*medians_i)/np.sum(omega_i0)
    tau2_nom = np.sum(omega_i0*pow((medians_i - median_0), 2)) - (k-1)
    tau2_den = np.sum(omega_i0) - np.sum(pow(omega_i0, 2))/np.sum(omega_i0)
    tau2 = np.max([0, tau2_nom/tau2_den])

    info('Computing federated median and its standard error')
    # Using approach from McGrath et al. (2019), section 2,
    # see: https://doi.org/10.1002/sim.8013
    omega_i = 1./(variances_i + tau2)
    federated_median = np.sum(medians_i*omega_i)/np.sum(omega_i)
    federated_median_std_err = np.sqrt(1./np.sum(omega_i))

    return {
        'value': federated_median,
        'std_err': federated_median_std_err
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
