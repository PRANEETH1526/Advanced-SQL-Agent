#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"


# Activate virtual environment
source .venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Run the Python app in the background with logging
nohup python src/run_service.py > app.log 2>&1 &

# Get the process ID and save it to a file
echo $! > app.pid

echo "App started in the background with PID: $(cat app.pid)"
echo "Logs: tail -f app.log"

