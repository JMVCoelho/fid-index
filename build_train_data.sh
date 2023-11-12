#!/bin/bash

#SBATCH --job-name=build_train_data

# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 60 # number cpus (threads) per task

# 327680
#SBATCH --mem=200000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

eval "$(conda shell.bash hook)"
conda activate openmatch

split=documents

tokenizer=./models/t5-base-scaled
negatives=./marco/$split/train.negatives.tsv
train_qrels=./marco/$split/qrels.train.tsv
train_queries=./marco/$split/train.query.txt
corpus=./marco/$split/corpus.tsv
save_folder=./marco/$split/processed_data/t5-base-marco-docs/

mkdir -p $save_folder

python OpenMatch/scripts/msmarco/build_train.py \
    --tokenizer_name $tokenizer \
    --negative_file $negatives  \
    --qrels $train_qrels  \
    --queries $train_queries  \
    --collection $corpus \
    --save_to $save_folder  \
    --doc_template "Title: <title> Text: <text>" \
    --n_sample 7


cd $save_folder
cat split*.jsonl > full.jsonl
rm split*.jsonl

line_count=$(wc -l full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val full.jsonl > val.jsonl
head -n $n_train full.jsonl > train.jsonl

rm full.jsonl