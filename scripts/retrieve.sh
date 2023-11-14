#!/bin/bash

#SBATCH --job-name=full_retrieval

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

#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:2

eval "$(conda shell.bash hook)"
conda activate openmatch

split=documents

model_name=t5-base-marco-$split-fusion
model_to_eval=./models/marco/$model_name
output_dir=./marco/$split/embeddings/$model_name
run_save=./marco/$split/results/$model_name

corpus=./marco/$split/corpus.tsv 
queries=./marco/$split/dev.query.txt
qrels=./marco/$split/qrels.dev.tsv

mkdir $run_save

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes 4 --multi_gpu OpenMatch/src/openmatch/driver/build_index.py  \
    --output_dir $output_dir \
    --model_name_or_path $model_to_eval \
    --per_device_eval_batch_size 6000  \
    --corpus_path $corpus \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1

/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes 2 --multi_gpu --main_process_port 30000 OpenMatch/src/openmatch/driver/retrieve.py  \
    --output_dir $output_dir  \
    --model_name_or_path $model_to_eval \
    --per_device_eval_batch_size 600  \
    --query_path $queries \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path $run_save/dev.trec \
    --dataloader_num_workers 1 \
    --use_gpu \
    --fusion 4

python OpenMatch/scripts/evaluate.py $qrels $run_save/dev.trec > $run_save/dev.results