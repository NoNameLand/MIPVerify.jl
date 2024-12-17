#!/bin/zsh

# Configuration
DOCKER="safe-pc-2.ef.technion.ac.il"
PORT=32778
PASSWORD='Ariel382$'  # Use single quotes for special characters
SOURCE="/GitProjects/MIPVerify.jl/"
DESTINATION="root@$DOCKER:/root/ERAN/ariel"

# Check if sshpass is installed
if ! command -v sshpass >/dev/null 2>&1; then
    echo "Error: sshpass not installed. Install it first."
    exit 1
fi

# Sync files to the remote server
echo "Syncing files to $DOCKER..."
sshpass -p "$PASSWORD" rsync -Crvz -e "ssh -p $PORT" "$SOURCE" "$DESTINATION"

# Check if rsync was successful
if [[ $? -eq 0 ]]; then
    echo "Files successfully synced to $DESTINATION."
else
    echo "Error: File sync failed."
    exit 1
fi
