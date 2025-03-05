from typing import Dict, List, Union
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import algorithm_client

from .v6_stats_utils import calculate_column_stats
from .v6_stats_utils import compute_federated_quantiles
from .v6_stats_utils import compute_local_quantile
from .v6_stats_utils import compute_local_quantile_sampling_variance
from .v6_stats_utils import compute_federated_mean
from .v6_stats_utils import compute_local_sum
from .v6_stats_utils import compute_local_nrows
from .v6_stats_utils import compute_local_sum_errors2
from .v6_stats_utils import compute_federated_counts
from .v6_stats_utils import compute_local_counts
from .v6_stats_utils import compute_federated_minmax
from .v6_stats_utils import compute_local_minmax
from .v6_stats_utils import compute_federated_nrows


@algorithm_client
def master(
    client: AlgorithmClient,
    statistics: Dict[str, List[str]],
    organization_ids: List[int] = None
) -> Dict[str, Dict[str, Union[int, Dict[str, Union[int, float]]]]]:
    """Compute simple statistics in a federated environment

    Parameters:
    - client: Vantage6 client object
    - statistics: Dictionary with columns and statistics to compute per column
    - organization_ids: Organization IDs to include, default includes all

    Returns:
    - Dictionary containing simple statistics per column
    """
    info('Collecting information on participating organizations')
    if not isinstance(organization_ids, list):
        organizations = client.organization.list()
        ids = [organization.get('id') for organization in organizations]
    else:
        ids = organization_ids

    info(f'Sending task to organizations {ids}')
    column_stats = calculate_column_stats(
        client=client,
        ids=ids,
        statistics=statistics
    )
    return column_stats
