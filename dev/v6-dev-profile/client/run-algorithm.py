#!/usr/bin/env python
# For local development. Launches a task that starts this algorithm on the
# local development/debugging network.
# At the moment, it simply launches partial method 'compute_local_stats' asking
# for mean and minmax statistics.

from vantage6.client import Client
from vantage6.cli.dev.profile import profile
from vantage6.testing.fixtures import wait_for_server, wait_for_nodes
import logging
import sys

# setup logging, as expected by vscode's task
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
# vscode's task we wrote (launch-task.problemMatcher) will look for this pattern
formatter = logging.Formatter("%(filename)s:%(lineno)d:0: %(levelname)s: %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)

task_config = {
    "collaboration": 1,
    "organizations": [1],
    "name": "local-dev-task",
    "image": "ghcr.io/maastrichtu-cds/v6-statistics-py:latest",
    "description": "Local development task",
    "databases": [{"label": "fakepublicdata"}, {"label": "fakepublicdata_config"}],
    "input_": {
        "method": "compute_local_stats",
        "kwargs": {
            "statistics": {
                "age": ["mean", "minmax"],
            }
        }
    }
}

profile_config = {
    "profile_name": "run-node-mars",
    "server_url": "http://127.0.6.1",
    "server_port": 80,
    "server_api": "/api",
    "username": "phobos",
    # just a test password for local dev, can be found in entities.yaml (OK to
    # be public)
    "password": "test-password-two-orbit",
    "organization_key": None,
}


def start_profile(profile_config):
    """
    Starts server and nodes on which the algorithm will run

    This specific collection of nodes and server is a v6 dev profile
    """
    try:
        profile(["start", profile_config["profile_name"]], standalone_mode=False)
    except SystemExit as e:
        if e.code != 0:
            log.error("Failed to start profile")
            raise RuntimeError("Failed to start profile")


def setup_client(profile_config) -> Client:
    log.info("Connecting to server")
    client = Client(
        profile_config["server_url"],
        profile_config["server_port"],
        profile_config["server_api"],
    )
    client.authenticate(profile_config["username"], profile_config["password"])
    client.setup_encryption(profile_config["organization_key"])
    return client


def send_task(client: Client, task_config: dict) -> dict:
    task = client.task.list(
        filters=[("name", task_config["name"]), ("status", "active")]
    )
    if task:
        log.warning("Active task found, killing it.")
        for t in task:
            client.task.delete(t["id"])
    log.info("Launching task")
    task = client.task.create(**task_config)
    return task


def main():
    log.info("Starting server and nodes...")
    start_profile(profile_config)

    log.info("Waiting for server to be ready at %s:%s%s...",
            profile_config["server_url"],
            profile_config["server_port"],
            profile_config["server_api"])
    server_wait = wait_for_server(
        profile_config["server_url"],
        profile_config["server_port"],
        profile_config["server_api"],
    )
    if server_wait:
        log.info("Server is ready")
    else:
        log.error("Server is not ready within the timeout")
        return

    log.info("Setting up client connection to the server...")
    client = setup_client(profile_config)

    log.info("Waiting for nodes to come online: %s", ["Mars - Planets Development Node"])
    wait_for_nodes(client, ["Mars - Planets Development Node"])

    log.info("Sending task to execute algorithm...")
    task = send_task(client, task_config)

    log.info("Waiting for results of task ID: %s", task["id"])
    client.wait_for_results(task["id"])

    log.info("Fetching results for task ID: %s", task["id"])
    results = client.result.from_task(task_id=task["id"])

    log.info("Results received: %s", results["data"][0]["result"])

if __name__ == "__main__":
    main()
