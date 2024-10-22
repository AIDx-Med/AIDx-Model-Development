#!/usr/bin/env bash

# setup rclone
LOCAL_RCLONE_CONFIG=$(realpath "$(dirname "$0")/rclone.conf")
TARGET_RCLONE_CONFIG="$HOME/.config/rclone/rclone.conf"

echo "Local rclone config: $LOCAL_RCLONE_CONFIG"
echo "Target rclone config: $TARGET_RCLONE_CONFIG"

# check if rclone is installed
if ! command -v rclone &> /dev/null
then
    echo "rclone could not be found. please install it."
    exit
fi

# check if rclone config file is in the same directory as the script regardless of where the script is called from
if [ ! -f "$LOCAL_RCLONE_CONFIG" ]; then
    echo "local rclone.conf not found, please place it in the same directory as the script"
    exit
fi


LOCAL_CONFIG_CONTENT=$(<"$LOCAL_RCLONE_CONFIG")

# Check if the target rclone configuration file exists
if [ -f "$TARGET_RCLONE_CONFIG" ]; then
    # Check if the entire content of the local configuration is in the target file
    if ! grep -qF "$LOCAL_CONFIG_CONTENT" "$TARGET_RCLONE_CONFIG"; then
        # If not present, append the local configuration to the target file
        echo "Appending new configuration to rclone config..."
        echo "$LOCAL_CONFIG_CONTENT" >> "$TARGET_RCLONE_CONFIG"
    else
        echo "Configuration already exists in rclone config."
    fi
else
    # If the target configuration file doesn't exist, copy the local one
    echo "Copying local configuration to rclone config..."
    mkdir -p "$(dirname "$TARGET_RCLONE_CONFIG")"
    cp "$LOCAL_RCLONE_CONFIG" "$TARGET_RCLONE_CONFIG"
fi