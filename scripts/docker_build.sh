#!/bin/bash

# Create a "tag" or name for the image
docker_tag=${USER}/dliris:latest

# Specify the base image for local use
docker_base=nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Build the Docker image
docker build .. -f ../docker/Dockerfile \
 --tag "${docker_tag}" \
 --build-arg DOCKER_BASE="${docker_base}" \
 --build-arg USER_ID="$(id -u)" \
 --build-arg GROUP_ID="$(id -g)" \
 --build-arg USER="${USER}" \
 --no-cache \
 --network=host

# Optionally, push the image to a registry
# docker push "${docker_tag}"
