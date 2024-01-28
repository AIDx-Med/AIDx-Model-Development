#!/usr/bin/env bash

# configure rclone
"$(dirname "$0")/utils/config_rclone.sh"

# query user for directory to download to
echo "Enter directory to download to: "
read -r DOWNLOAD_DIR

# query user for remote directory to download from
echo "Enter remote directory to download from: "
read -r REMOTE_DIR

# make the download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

DOWNLOAD_DIR=$(realpath "$DOWNLOAD_DIR")

# download from google drive
rclone copy "gdrive:/$REMOTE_DIR" "$DOWNLOAD_DIR" -P -v