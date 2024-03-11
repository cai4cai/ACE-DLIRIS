#!/bin/bash

# Read each line from batch_jobs.txt and submit a job
while IFS=' ' read -r model data sys mode gpu_memory; do
  # Skip lines starting with '#'
  [[ "$model" == \#* ]] && continue

  # Default GPU memory to 32G if not provided
  if [ -z "$gpu_memory" ]; then
    gpu_memory="32G"
  fi

  # Submit the job
  bash runai_submit.sh "${model}" "${data}" "${sys}" "${mode}" "${gpu_memory}"
done < batch_jobs.txt
