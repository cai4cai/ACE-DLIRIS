#!/bin/bash

# Usage example: ./docker_run.sh --mode train --sys high --data brats_2021 --model baseline_ce [--gpu <gpu_number>] [--cpus <cpus>] [--shm-size <shm_size>]

# Get the directory of the current script
SCRIPT_DIR="$(dirname "$0")"

# Navigate to the parent directory of the script, which is assumed to be the project root
PROJECT_DIR="$(realpath "$SCRIPT_DIR/..")"

# Define the data directory relative to the project directory
# Assuming the data directory is at the same level as the ace_dliris project directory
DATA_DIR="$(realpath "$PROJECT_DIR/../data")"

# Docker image name
IMAGE_NAME="${USER}/dliris:latest"

# Default values for optional arguments
GPU=0         # Default GPU index
CPUS="0-5"    # Default CPU range
SHM_SIZE="32g" # Default shared memory size

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      GPU="$2"
      shift # past argument
      shift # past value
      ;;
    --cpus)
      CPUS="$2"
      shift # past argument
      shift # past value
      ;;
    --shm-size)
      SHM_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL_ARGS+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo "Using GPU: $GPU, CPUs: $CPUS, SHM_SIZE: $SHM_SIZE"
echo "Arguments to run_monai_bundle.py: $@"

# Run the Docker container with the configured arguments
docker run --rm \
    --gpus '"device='$GPU'"' \
    --cpuset-cpus=$CPUS \
    --shm-size=$SHM_SIZE \
    --volume $PROJECT_DIR:/workspace/project \
    --volume $DATA_DIR:/workspace/data \
    --workdir /workspace/project \
    $IMAGE_NAME \
    python ./run_monai_bundle.py "$@"

# Explanation of options:
# --rm: Automatically remove the container when it exits
# --gpus: Specify which GPU to use
# --cpuset-cpus: Specify which CPUs to use
# --shm-size: Specify the shared memory size for the container
# --volume: Map volumes for project and data directories
# --workdir: Set working directory inside the container
# "$@": Pass all remaining script arguments to the Python script inside the container
