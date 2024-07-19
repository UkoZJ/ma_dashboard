#!/bin/bash

set -e

micromamba activate ccaes_dashboard

#pip install -r requirements.txt

export OPTS="$OPTS --allow-websocket-origin=*"
export OPTS="$OPTS --warm"
export OPTS="$OPTS --reuse-sessions"
export OPTS="$OPTS --global-loading-spinner"
export OPTS="$OPTS --port 5006"

# Enable options for authentication
# export OPTS="$OPTS --static-dirs assets=./assets"
# export OPTS="$OPTS --basic-auth .credentials.json"
# export OPTS="$OPTS --cookie-secret admin_pass"
# export OPTS="$OPTS --basic-login-template ./panel_apps/login.html"

python -m panel serve $OPTS panel_apps/english.py