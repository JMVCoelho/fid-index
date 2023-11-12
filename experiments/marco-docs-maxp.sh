#!/bin/bash

#SBATCH --job-name=marco-docs-maxp

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
text_length=128
n_passages=4

n_gpus=4

#python data_scripts/marco_documents/build_data_maxp.py $text_length $n_passages

initial_model=./models/t5-base-scaled

first_trained_model_name=t5-base-marco-$split-maxp

negatives=./marco/$split/train.negatives.tsv

train_qrels=./marco/$split/qrels.train.tsv
train_queries=./marco/$split/train.query.filtered.txt
corpus=./marco/$split/corpus_${n_passages}p_$text_length.tsv


initial_data_save_folder=./marco/$split/processed_data/$first_trained_model_name

#mkdir -p $save_folder
#
#echo "########################################"
#echo "Building initial data"
#echo "########################################"
#
#python OpenMatch/scripts/msmarco/build_train.py \
#    --tokenizer_name $initial_model \
#    --negative_file $negatives  \
#    --qrels $train_qrels  \
#    --queries $train_queries  \
#    --collection $corpus \
#    --save_to $initial_data_save_folder  \
#    --doc_template "Title: <title> Text: <text>" \
#    --n_sample 7 \
#    --split_sentences "[PSEP]"
#
#cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/full.jsonl
#rm $initial_data_save_folder/split*.jsonl
#
#line_count=$(wc -l $initial_data_save_folder/full.jsonl | awk '{print $1}')
#n_val=500
#n_train=$((line_count - n_val))
#
#echo $n_train
#
#tail -n $n_val $initial_data_save_folder/full.jsonl > $initial_data_save_folder/val.jsonl
#head -n $n_train $initial_data_save_folder/full.jsonl > $initial_data_save_folder/train.jsonl
#
#rm $initial_data_save_folder/full.jsonl

# Train model with initial negatives
first_model_output_path=./models/marco/$first_trained_model_name
train_data=$initial_data_save_folder/train.jsonl
valid_data=$initial_data_save_folder/val.jsonl

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes $n_gpus --multi_gpu OpenMatch/src/openmatch/driver/train_dr.py  \
    --output_dir $first_model_output_path \
    --model_name_or_path $initial_model \
    --do_train \
    --save_steps 300  \
    --eval_steps 300  \
    --train_path $train_data  \
    --eval_path $valid_data  \
    --fp16 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 8 \
    --train_n_passages 8  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len $text_length  \
    --num_train_epochs 3  \
    --report_to wandb \
    --logging_steps 10 \
    --run_name $first_trained_model_name \
    --evaluation_strategy steps \
    --dataloader_num_workers 4 \
    --maxp 4

