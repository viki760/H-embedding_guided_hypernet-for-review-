#!/bin/bash

# Get the current date
current_date=$(date +%Y-%m-%d)

# Create a directory with the current date
mkdir -p result_$current_date

python train_resnet.py --use_adam --custom_network_init --plateau_lr_scheduler --lambda_lr_scheduler --epochs 1 --emb_reg --emb_metric Hembedding --temb_size 24 --lr 0.0005 --emb_beta 0.05 --random_seed 42
