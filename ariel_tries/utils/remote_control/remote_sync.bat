#!/bin/zsh

# Configuration
DOCKER="safe-pc-2.ef.technion.ac.il"
PORT=32778
PASSWORD='LondonCookies42$'  # Use single quotes for special characters


# Check if sshpass is installed
if ! command -v sshpass >/dev/null 2>&1; then
    echo "Error: sshpass not installed. Install it first."
    exit 1
fi

# Sync files to the remote server
echo "Syncing files to $DOCKER..."
shpass -p PASSWORD ssh -o StrictHostKeyChecking=no DOCKER