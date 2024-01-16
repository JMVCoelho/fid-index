# Given the data from here: https://github.com/thunlp/OpenMatch/blob/fbf198ad5353e35b8b67ea7539783db61b32d1f2/v1/docs/experiments-msmarco-doc.md
# build files files in normalized format


import re

import pickle

import sys

MAX_DOC_LEN = int(sys.argv[1]) # 128
N_PASSAGES = int(sys.argv[2]) # 4

corpus_only = True

def is_valid_id(did):
    pattern = re.compile(r'^D\d+$')
    return bool(pattern.match(did))

ID_MAPPER_DOCS = {}

import re

def split_text_into_passages(text, max_words_per_passage, n_passages):
    # Remove extra spaces and split the text into sentences

    # give some slack
    max_words_per_passage += 10

    sentences = re.split(r'(?<=[.!?])', text)

    # Initialize passages
    passages = ['']

    # Iterate through sentences and add them to passages
    for sentence in sentences:
        # Remove extra spaces and split the sentence into words
        words = re.findall(r'\S+', sentence)
        if len(words) > max_words_per_passage:
            # Split long sentences into smaller segments
            sub_sentences = [words[i:i+max_words_per_passage] for i in range(0, len(words), max_words_per_passage)]
            for sub_sentence in sub_sentences:
                passage = ' '.join(sub_sentence)
                if len(passages[-1].split()) + len(sub_sentence) > max_words_per_passage:
                    # Start a new passage if adding this sentence would exceed the word limit
                    passages.append('')
                passages[-1] += ' ' + passage.strip()
        else:
            if len(passages[-1].split()) + len(words) > max_words_per_passage:
                # Start a new passage if adding this sentence would exceed the word limit
                passages.append('')
            passages[-1] += ' ' + sentence.strip()

    # Trim leading space from each passage
    passages = [passage.strip() for passage in passages]

    # Ensure there are always exactly four passages
    while len(passages) < n_passages:
        passages += passages

    return passages[:n_passages] 


with open("/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/msmarco-docs.tsv", "r") as f, \
    open(f"/user/home/jcoelho/clueweb-structured-retrieval/marco/documents/corpus_{N_PASSAGES}p_{MAX_DOC_LEN}_agg.tsv", "w") as fout:
    start_id = 0
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
            did = did[1:]
        else:
            raise Exception("problem with ids: corpus")
        
        ID_MAPPER_DOCS[did] = str(start_id)

        text = text.replace("\"", "").replace("\'", "").replace("\n", "").replace("\t", "").replace("\r", "")

        passages = split_text_into_passages(text, MAX_DOC_LEN, N_PASSAGES)

        aggregator = []

        for passage in passages:
            aggregator.append(" ".join(passage.split(" ")[:int((MAX_DOC_LEN/N_PASSAGES)-7)]))

        aggregated_string = " ".join(aggregator)

        passages = [aggregated_string] + passages

        text = "[PSEP]".join(passages)

        assert len(text.split('[PSEP]')) == N_PASSAGES + 1

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