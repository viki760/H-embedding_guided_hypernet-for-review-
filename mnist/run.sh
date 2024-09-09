#!/bin/bash
conda init
conda activate hypernet

# Get the current date
current_date=$(date +%Y-%m-%d)

# Create a directory with the current date
mkdir -p tuning_$current_date

# Define an array of learning rates
learning_rates=(0.001 0.01 0.1)
beta_reg=(0.02 0.05 0.1 0.2 0.5 1.0 1.5 2.0)


# Loop through the arguments
for beta in "${beta_reg[@]}"
do
# Run train.py with the current arguments
python train_permutedMNIST.py --emb_beta=$beta > "tuning_${current_date}/result_Hemb_beta_${beta}.txt"
done
