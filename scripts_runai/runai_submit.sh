#!/bin/bash

# Construct job name from mode, model, and data
job_name="${4}-${1}-${2}"

# Replace underscores with dashes in job name
job_name=$(echo "${job_name}" | tr '_' '-')

runai delete job tb-"${job_name}"

# Use the GPU memory value in the runai submit command
runai submit tb-"${job_name}" \
  -i aicregistry:5000/${USER}/ace_dliris \
  -p tbarfoot \
  --gpu-memory "${5}" \
  --host-ipc \
  -v /nfs:/nfs \
  -e MPLCONFIGDIR="/tmp/" \
  -- bash /nfs/home/${USER}/ace_dliris/scripts_runai/startup.sh "$1" "$2" "$3" "$4"
