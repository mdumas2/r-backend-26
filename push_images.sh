#!/usr/bin/env bash
set -e

# Change this if your dockerhub username is different
DOCKER_USER="mdumasl"

echo "Building API image for linux/amd64..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t ${DOCKER_USER}/r-backend-26-api:latest -f Dockerfile .

echo "Building LB image for linux/amd64..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t ${DOCKER_USER}/r-backend-26-lb:latest -f Dockerfile.lb .

echo "Pushing images to Docker Hub..."
docker push ${DOCKER_USER}/r-backend-26-api:latest
docker push ${DOCKER_USER}/r-backend-26-lb:latest

echo "Done! The images are now public."
