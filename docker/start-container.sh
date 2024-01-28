#!/usr/bin/env bash

# get "build" argument
BUILD=$1

# if build is present, add --build to docker compose command
if [ "$BUILD" = "--build" ]; then
    docker compose build --no-cache aidx-jupyter
    docker compose up -d
    exit
fi

docker compose up -d
