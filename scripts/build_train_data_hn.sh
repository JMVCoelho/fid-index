#!/bin/bash

#SBATCH --job-name=build_train_data_hn

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

python OpenMatch/scripts/msmarco/build_hn.py  \
    --tokenizer_name ./models/t5-base-scaled  \
    --hn_file ./marco/passage/negatives/t5-base-marco/train.trec  \
    --qrels ./marco/passage/qrels.train.tsv  \
    --queries ./marco/passage/train.query.txt  \
    --collection ./marco/passage/corpus.tsv  \
    --save_to ./marco/passage/processed_data/t5-base-marco-self-hn-1/  \
    --doc_template "Title: <title> Text: <text>" \
    --n_sample 7

cd ./marco/passage/processed_data/t5-base-marco-self-hn-1
cat *.hn.jsonl > full.jsonl
rm *.hn.jsonl

line_count=$(wc -l full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val full.jsonl > val.jsonl
head -n $n_train full.jsonl > train.jsonl

rm full.jsonl

