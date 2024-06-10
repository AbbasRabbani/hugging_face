#!/bin/bash

# Check if port argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <port>"
  exit 1
fi

# Assign the port argument to a variable
PORT=$1

# Build the Docker image, passing the port as a build argument
docker build --build-arg PORT=${PORT} -t huggingface-app .

# Run the Docker container with the specified port
docker run -p ${PORT}:${PORT} huggingface-app
