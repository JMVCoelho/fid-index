# Given the data from here: https://github.com/thunlp/OpenMatch/blob/fbf198ad5353e35b8b67ea7539783db61b32d1f2/v1/docs/experiments-msmarco-doc.md
# build files files in normalized format

# this uses a tokenizer instead of sentence level segmentation.

import re

import pickle

import sys

from transformers import AutoTokenizer

from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("/user/home/jcoelho/clueweb-structured-retrieval/models/t5-base-scaled")

MAX_DOC_LEN = sys.argv[1] # 512
N_PASSAGES = sys.argv[2] # 4

corpus_only = True

def is_valid_id(did):
    pattern = re.compile(r'^D\d+$')
    return bool(pattern.match(did))

ID_MAPPER_DOCS = {}

import re

def split_list(lst, n):
    quotient = len(lst) // n
    remainder = len(lst) % n

    divided_list = [lst[i * quotient + min(i, remainder):(i + 1) * quotient + min(i + 1, remainder)] for i in range(n)]
    
    return divided_list


with open("/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/msmarco-docs.tsv", "r") as f, \
    open(f"/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/corpus_{N_PASSAGES}p_{MAX_DOC_LEN}.tsv", "w") as fout:
    start_id = 0
    for line in tqdm(f, total=3213835):
        try:
            did, url, title, text = line.strip().split("\t")
        except Exception:
            if len(line.strip().split("\t")) == 3:
                did = line.strip().split("\t")[0]
                title = "-"
                text = line.strip().split("\t")[1]
            if len(line.strip().split("\t")) == 2:
                did = line.strip().split("\t")[0]
                title = "-"
                text = line.strip().split("\t")[-1]
            if len(line.strip().split("\t")) == 1:
                raise Exception("this should not happen if initial dataset is correct, check it")
        if is_valid_id(did):
            did = did[1:]
        else:
            raise Exception("problem with ids: corpus")
        
        ID_MAPPER_DOCS[did] = str(start_id)

        text = text.replace("\"", "").replace("\'", "").replace("\n", "").replace("\t", "").replace("\r", "")
        
        tokenized_text = tokenizer.encode(text)[:-1]

        splits = split_list(tokenized_text, int(N_PASSAGES))

        split_text = tokenizer.batch_decode(splits)

        text = '[PSEP]'.join(split_text)

        assert len(text.split('[PSEP]')) == int(N_PASSAGES)

        fout.write(f"{ID_MAPPER_DOCS[did]}\t{title}\t{text}\n")
        start_id += 1

if corpus_only:
    exit(0)


# train.negatives.tsv <qid>\t<neg1,neg2,neg3...>
negatives = {}
with open("marco/documents/bids_marco-doc_ance-maxp-10.tsv", 'r') as f:
    for line in f:
        qid, did, label = line.strip().split()
        if label == "1":
            continue
        else:
            if is_valid_id(did):
                did = ID_MAPPER_DOCS[did[1:]]
            else:
                raise Exception("problem with ids")

            if qid not in negatives:
                negatives[qid] = [did] 
                
            else: 
                negatives[qid].append(did)

with open("marco/documents/train.negatives.tsv", 'w') as fout:
    for qid in negatives:
        negs = ",".join(negatives[qid])

        fout.write(f"{qid}\t{negs}\n")



with open("marco/documents/msmarco-doctrain-qrels.tsv", 'r') as f, \
    open("marco/documents/qrels.train.tsv", 'w') as fout:

    for line in f:
        qid, z, did, rel = line.strip().split()

        if qid in negatives:
            if is_valid_id(did):
                did = ID_MAPPER_DOCS[did[1:]]
            else:
                raise Exception("problem with ids: train qrels")
        
            fout.write(f"{qid}\t{z}\t{did}\t{rel}\n")

with open("marco/documents/msmarco-docdev-qrels.tsv", 'r') as f, \
    open("marco/documents/qrels.dev.tsv", 'w') as fout:

    for line in f:
        qid, z, did, rel = line.strip().split()

        if is_valid_id(did):
            did = ID_MAPPER_DOCS[did[1:]]
        else:
            raise Exception("problem with ids: dev qrels")
    
        fout.write(f"{qid}\t{z}\t{did}\t{rel}\n")


with open("/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/msmarco-docs.tsv", "r") as f, \
    open("/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/corpus.tsv", "w") as fout:

    for line in f:
        try:
            did, url, title, text = line.strip().split("\t")
        except Exception:
            if len(line.strip().split("\t")) == 3:
                did = line.strip().split("\t")[0]
                title = "-"
                text = line.strip().split("\t")[1]
            if len(line.strip().split("\t")) == 2:
                did = line.strip().split("\t")[0]
                title = "-"
                text = line.strip().split("\t")[-1]
            if len(line.strip().split("\t")) == 1:
                raise Exception("this should not happen if initial dataset is correct, check it")
        if is_valid_id(did):
            did = ID_MAPPER_DOCS[did[1:]]
        else:
            raise Exception("problem with ids: corpus")

        text = text.replace("\"", "").replace("\'", "").replace("\n", "").replace("\t", "").replace("\r", "")
        fout.write(f"{did}\t{title}\t{text}\n")

        
with open('marco/documents/id2seqint_mapper.tsv', 'wb') as h:
    pickle.dump(ID_MAPPER_DOCS, h, protocol=pickle.HIGHEST_PROTOCOL)