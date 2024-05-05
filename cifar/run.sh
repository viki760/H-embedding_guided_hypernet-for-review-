#!/bin/bash

conda activate hypernet

# Define the arguments for train.py
args=(["arg1","arg2"] ["arg3","arg4"] ["arg5","arg6"])

# Get the current date
current_date=$(date +%Y-%m-%d)

# Create a directory with the current date
mkdir -p $current_date

# Loop through the arguments
for arg_pair in "${args[@]}"
do
    # Split the pair into two arguments
    IFS=',' read -ra arg <<< "$arg_pair"

    # Run train.py with the current arguments
    python train.py ${arg[0]} ${arg[1]} > "${current_date}/result_${arg[0]}_${arg[1]}.txt"
done