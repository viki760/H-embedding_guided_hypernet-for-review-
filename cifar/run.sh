#!/bin/bash
# conda init
conda activate hypernet

# Get the current date
current_date=$(date +%Y-%m-%d)

# Create a directory with the current date
mkdir -p $current_date

# Loop through the arguments
while IFS= read -r line
do
    # Run train.py with the current arguments
    python train_fixed_emb.py $line > "${current_date}/result_fixed_emb_${line// /_}.txt"
done < args/args.txt