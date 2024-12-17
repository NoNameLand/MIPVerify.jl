#!/bin/zsh

# Configuration
DOCKER="safe-pc-2.ef.technion.ac.il"
PORT=32778
PASSWORD='Ariel382$'  # Use single quotes for special characters
REMOTE_SCRIPT="/root/ERAN/ariel/run_script.sh"  # Path to the file you want to run remotely

# Check if sshpass is installed
if ! command -v sshpass >/dev/null 2>&1; then
    echo "Error: sshpass not installed. Install it first."
    exit 1
fi

# Execute the remote script
echo "Running $REMOTE_SCRIPT on $DOCKER..."
sshpass -p "$PASSWORD" ssh -p $PORT "root@$DOCKER" "bash $REMOTE_SCRIPT"

# Check if the remote script ran successfully
if [[ $? -eq 0 ]]; then
    echo "Script executed successfully on $DOCKER."
else
    echo "Error while running script on remote server."
    exit 1
fi
