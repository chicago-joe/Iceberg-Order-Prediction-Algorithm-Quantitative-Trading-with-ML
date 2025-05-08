#!/bin/bash

# Set the port for our local Jupyter process\np
port="9090"

# Define environment variables that will be used by MyST
# We'll use the values of these variables in our Jupyter server as well.
export JUPYTER_BASE_URL="http://localhost:${port}"
export JUPYTER_TOKEN="1234"

uv run --no-cache jupyter-server --allow-root --ip 0.0.0.0 --port 9090 --IdentityProvider.token='1234' --ServerApp.allow_origin='*' &
echo $JUPYTER_BASE_URL

