# Simple statistics with vantage6

This repository contains an implementation of basic statistics 
designed for federated learning environments via the vantage6 framework. 
The following operations are supported: counts, min, max, mean and quantiles.

Follow the instructions in subsequent sections to set up and execute the 
federated statistics analysis.


## Usage

This section provides a comprehensive guide on how to use the repository to 
perform federated statistics analysis, from initialising the client to 
executing the task and retrieving the results.

1. **Install the vantage6 client**
```bash
pip install vantage6-client
```

2. **Initialise the vantage6 client**
```python
from vantage6.client import Client

# Load your configuration settings from a file or environment
config = {
    'server_url': '<API_ENDPOINT>',
    'server_port': <API_PORT>,
    'server_api': '<API_VERSION>',
    'username': '<USERNAME>',
    'password': '<PASSWORD>',
    'organization_key': '<ORGANIZATION_PRIVATE_KEY>'
}

client = Client(config['server_url'], config['server_port'], config['server_api'])
client.authenticate(username=config['username'], password=config['password'])
client.setup_encryption(config['organization_key'])
```

Replace the placeholders in `config` with your actual configuration details.

3. **Define the algorithm input**
```python
input_ = {
    'method': 'master',
    'kwargs': {
        'statistics': {
            'columnA': ['mean', 'minmax', 'quantiles'],
            'columnB': ['counts']
        }, # Define which statistics to compute per column
        'organization_ids': [1, 2, 3] # Example organization IDs
    }
}
```

Specify organization IDs and which statistics to be computed per column.

4. **Create and run the task**
```python
task = client.task.create(
    collaboration=3,  # Use your specific collaboration ID
    organizations=[1, 2, 3],  # List your organization IDs
    name='simple_statistics',  # Give your task a specific name
    image='ghcr.io/mdw-nl/v6-statistics:v1',  # Specify the desired algorithm Docker image version
    description='Simple statistics',  # Describe the task
    databases=[{'label': 'my_database_label'}],  # Specify your database label
    input_=input_
)
```

Provide actual values for the `collaboration`, `organizations`, `name`, `image`, 
`description`, and `databases` fields.

5. **Monitor and retrieve results**: Use the vantage6 client methods to check 
    the status of the task and retrieve the results when the task is complete.

