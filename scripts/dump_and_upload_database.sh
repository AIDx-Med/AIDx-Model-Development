#!/usr/bin/env bash

# This script dumps the database and uploads it to google drive

"$(dirname "$0")/utils/config_rclone.sh"

# query user for chunk size, test size, and parquet directory
echo "Enter chunk size (amount of rows to process at a time): "
read -r CHUNK_SIZE

echo "Enter test-dataset percent (as a decimal): "
read -r TEST_SIZE

echo "Enter parquet directory: "
read -r PARQUET_DIR

# dump database

# change directory to the root project directory
cd "$(dirname "$0")/.."

# run the dump script
python aidx.py create-parquet-datasets --chunk-size "$CHUNK_SIZE" --test-size "$TEST_SIZE" --parquet-dir "$PARQUET_DIR"

PARQUET_DIR=$(realpath "$PARQUET_DIR")

# upload to google drive
rclone copy "$PARQUET_DIR" "gdrive:/aidx-parquet-datasets" -P -v