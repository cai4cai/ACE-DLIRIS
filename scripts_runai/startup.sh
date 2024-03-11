#!/bin/bash
echo $USER
echo $HOME
echo $PWD

export PATH=/${HOME}/miniconda3/bin/:$PATH
cd /nfs/${HOME}/ace_dliris/ || exit

# Pass the arguments to the Python script
python run_monai_bundle.py --model "${1}" --data "${2}" --sys "${3}" --mode "${4}"
