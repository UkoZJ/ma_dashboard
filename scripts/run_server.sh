#!/bin/bash

set -e

# Set the path to your virtual environment's Python executable
export VENV_PATH="/home/uko/Dev/ma_dashboard/.envs/ma_dashboard_env"

# Activate the micromamba environment
micromamba activate "$VENV_PATH"

# Uncomment this line if you need to install requirements
# pip install -r requirements.txt

# Set options for the panel server
export OPTS="$OPTS --allow-websocket-origin=*"
export OPTS="$OPTS --warm"
export OPTS="$OPTS --reuse-sessions"
export OPTS="$OPTS --global-loading-spinner"
export OPTS="$OPTS --port 5006"

# Enable options for authentication if needed
# export OPTS="$OPTS --static-dirs assets=./assets"
# export OPTS="$OPTS --basic-auth .credentials.json"
# export OPTS="$OPTS --cookie-secret admin_pass"
# export OPTS="$OPTS --basic-login-template ./panel_apps/login.html"

# Start the panel server with the specified options
python -m panel serve $OPTS panel_apps/english.py