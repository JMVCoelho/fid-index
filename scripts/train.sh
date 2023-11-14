#!/bin/bash

#SBATCH --job-name=t5-base-marco

# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12 # number cpus (threads) per task

# 327680
#SBATCH --mem=200000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

# You can also change the number of requested GPUs
# replace the XXX with nvidia_a100-pcie-40gb or nvidia_a100-sxm4-40gb
# replace the YYY with the number of GPUs that you need, 1 to 8 PCIe or 1 to 4 SXM4

#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:4

eval "$(conda shell.bash hook)"
conda activate openmatch

split=documents

model_to_train=./models/t5-base-scaled
run_name=t5-base-marco-$split
output_path=./models/marco/$run_name
train_data=./marco/$split/processed_data/$run_name/train.jsonl
valid_data=./marco/$split/processed_data/$run_name/val.jsonl

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes 4 --multi_gpu OpenMatch/src/openmatch/driver/train_dr.py  \
    --output_dir $output_path \
    --model_name_or_path $model_to_train \
    --do_train \
    --save_steps 300  \
    --eval_steps 300  \
    --train_path $train_data  \
    --eval_path $valid_data  \
    --fp16 \
    --per_device_train_batch_size 40 \
    --gradient_accumulation_steps 2 \
    --train_n_passages 8  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 3  \
    --report_to wandb \
    --logging_steps 10 \
    --run_name $run_name \
    --evaluation_strategy steps \
    --dataloader_num_workers 4