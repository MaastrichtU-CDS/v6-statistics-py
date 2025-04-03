#!/bin/bash
# starts the algorithm via debugpy to allow for remote debugging

set -euo pipefail

log() {
    echo "[$(basename $0) | $(date +'%Y-%m-%dT%H:%M:%S%z')]: $*"
}

# debugpy: host, port, and path to debugpy package
debugpy_host="0.0.0.0"
debugpy_port="5678"
debugpy_path="$V6_ALGORITHM_DEBUG_DEBUGGER_DIR/debugpy"


log "Starting in debugging mode with debugpy.."
if [[ ! -d "${debugpy_path}" ]]; then
    log "Could not find debugpy at ${debugpy_path}!"
    log "If running within a container, was the path set correctly?"
    exit 1
fi

# typically, the algorithm python package is already installed (/usr/..) we
# override this with the local package so /app code is actually used
# TODO: this is rather clunky..
pip install -e /app

# we need to tell python where to find the debugpy package
export PYTHONPATH="${debugpy_path}"

log "Will now wait for a debugger to attach on ${debugpy_host}:${debugpy_port} [debugpy]!"
# `--wait-for-client` can be passed to debugpy and it will wait for a debugger
# to attach before starting vnode-local
python3 -m debugpy \
    --wait-for-client \
    --listen "${debugpy_host}:${debugpy_port}" \
    -c "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"
