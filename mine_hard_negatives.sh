#!/bin/bash

#SBATCH --job-name=mine_hn

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

export WORLD_SIZE=4
#gpu node level id
export LOCAL_RANK=0,1,2,3
#gpu cluster level id
export RANK=0,1,2,3

split=documents

model_name=t5-base-marco-$split-self-hn-1
model=./models/marco/$model_name
output_dir=./marco/$split/embeddings/$model_name
run_save=./marco/$split/negatives/$model_name


#corpus=./marco/$split/corpus_firstp_128.tsv 
queries=./marco/$split/train.query.filtered.txt
qrels=./marco/$split/qrels.train.tsv

negatives_save_folder=./marco/$split/negatives/$model_name/

mkdir $run_save


#/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes 4 --multi_gpu OpenMatch/src/openmatch/driver/build_index.py  \
#    --output_dir $output_dir \
#    --model_name_or_path $model  \
#    --per_device_eval_batch_size 6000  \
#    --corpus_path $corpus  \
#    --doc_template "Title: <title> Text: <text>"  \
#    --doc_column_names id,title,text  \
#    --q_max_len 32  \
#    --p_max_len 128  \
#    --fp16  \
#    --dataloader_num_workers 4
#
#echo "search"
#
#/home/jcoelho/.conda/envs/openmatch/bin/accelerate launch --num_processes 4 --multi_gpu OpenMatch/src/openmatch/driver/retrieve.py  \
#    --output_dir $output_dir  \
#    --model_name_or_path $model  \
#    --per_device_eval_batch_size 600  \
#    --query_path $queries  \
#    --query_template "<text>"  \
#    --query_column_names id,text  \
#    --q_max_len 32  \
#    --fp16  \
#    --trec_save_path $run_save/train.trec \
#    --dataloader_num_workers 4 \
#    --use_gpu
#
#echo "building files"


python OpenMatch/scripts/msmarco/build_hn.py  \
    --tokenizer_name $model  \
    --hn_file $run_save/train.trec \
    --qrels $qrels \
    --queries $queries  \
    --collection $corpus  \
    --save_to $negatives_save_folder  \
    --doc_template "Title: <title> Text: <text>" \
    --n_sample 7

cd $negatives_save_folder
cat *.hn.jsonl > full.jsonl
rm *.hn.jsonl

line_count=$(wc -l full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val full.jsonl > val.jsonl
head -n $n_train full.jsonl > train.jsonl

rm full.jsonl