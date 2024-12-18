#!/bin/zsh

# Configuration
DOCKER="safe-pc-2.ef.technion.ac.il"
PORT=32778
PASSWORD='LondonCookies42$'  # Use single quotes for special characters

# Check if sshpass is installed
if ! command -v sshpass >/dev/null 2>&1; then
    echo "Error: sshpass is not installed. Please install it first."
    exit 1
fi

# Sync files to the remote server
echo "Attempting to sync files to $DOCKER..."
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p "$PORT" "$DOCKER" exit

# Check the exit status of the SSH command
if [ $? -eq 0 ]; then
    echo "Success: Synced with $DOCKER."
else
    echo "Failure: Unable to sync with $DOCKER."
fi

# Wait for user input before exiting
echo "Press any key to exit..."
read -n 1 -s