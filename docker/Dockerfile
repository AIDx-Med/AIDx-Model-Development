# Use lambda-stack as base image
FROM lambda-stack:22.04

# Set the working directory in the container
WORKDIR /workspace

# Install Packages
COPY ../requirements.txt .
RUN pip install --force-reinstall -r requirements.txt