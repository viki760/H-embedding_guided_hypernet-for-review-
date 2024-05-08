#!/bin/bash

conda activate hypernet

# Get the current date
current_date=$(date +%Y-%m-%d)

# Create a directory with the current date
mkdir -p $current_date

# Loop through the arguments
while IFS= read -r line
do
    # Run train.py with the current arguments
    python train.py $line > "${current_date}/result_${line// /_}.txt"
done < args/args.txt