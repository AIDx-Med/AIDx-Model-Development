docker stop docker-aidx-jupyter-1
docker rm docker-aidx-jupyter-1

# get "build" argument
BUILD=$1

# if build is present, add --build to docker compose command
if [ "$BUILD" = "--build" ]; then
    ./start-container.sh --build
    exit
fi

./start-container.sh
