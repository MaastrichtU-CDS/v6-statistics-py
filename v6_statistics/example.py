import json
import warnings
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
warnings.filterwarnings('ignore')


# Initialize the mock server. The datasets simulate the local datasets from
# the node. In this case we have two parties having two different datasets:
# a.csv and b.csv. The module name needs to be the name of your algorithm
# package. This is the name you specified in `setup.py`, in our case that
# would be v6-correlation-matrix-py.
dataset_1 = {'database': './v6_statistics/local/data1.csv', 'db_type': 'csv'}
dataset_2 = {'database': './v6_statistics/local/data2.csv', 'db_type': 'csv'}
dataset_3 = {'database': './v6_statistics/local/data3.csv', 'db_type': 'csv'}
org_ids = ids = [0, 1, 2]

client = MockAlgorithmClient(
    datasets = [[dataset_1], [dataset_2], [dataset_3]],
    organization_ids=org_ids,
    module='v6_statistics'
)

organizations = client.organization.list()
org_ids = ids = [organization['id'] for organization in organizations]

# To trigger the master method you also need to supply the `master`-flag
# to the input. Also note that we only supply the task to a single organization
# as we only want to execute the central part of the algorithm once. The master
# task takes care of the distribution to the other parties.
average_task = client.task.create(
    input_={
        'method': 'master',
        'kwargs': {
            'statistics': {
                'columnA': ['mean', 'minmax', 'quantiles'],
                'columnB': ['counts']
            }
        }
    },
    organizations=[org_ids[0]]
)

results = client.result.get(average_task.get('id'))
print(json.dumps(results, indent=4))
