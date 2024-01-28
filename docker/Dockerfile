# Use lambda-stack as base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /workspace

ADD . .

# Install environment
SHELL ["/bin/bash", "-c"]