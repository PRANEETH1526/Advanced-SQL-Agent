#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

# Stop the app if running
if [[ -f app.pid ]]; then
    ./stop.sh
fi

# Start the app again
./start.sh

