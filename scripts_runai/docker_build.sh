#!/bin/bash
# Create a "tag" or name for the image
docker_tag=aicregistry:5000/${USER}/ace_dliris

docker_base=nvidia/cuda:11.8.0-runtime-ubuntu22.04

docker build . -f ../docker/Dockerfile \
 --tag "${docker_tag}" \
 --build-arg DOCKER_BASE="${docker_base}" \
 --build-arg USER_ID="$(id -u)" \
 --build-arg GROUP_ID="$(id -g)" \
 --build-arg USER="${USER}" \
 --network=host

docker push "${docker_tag}"