# Processing of 1B Words Language Model for research project

This repo contains scripts to dump load the trained langauge model embeddings and dump them as numpy readable files

how to use:

## 1. download_language_model.sh
Downloads all necessary files

## 2. generate conll vocab file with build_data.py in https://github.com/MissMuffin/sequence_tagging/tree/lm

## 3. config.py
set all file paths here

## 4. dump_emb.py
Can:
1. dump softmax to numpy file
2. dump language model embeddings to numpy or text file
    - non ascii will be removed
    - number of dumped embeddings can be limited with vocab_length 

## 5. trimm_emb.py
### def build_trimmed_emb
- set in config.py)
- will load "raw" embeddings from file (non ascii removed)
- will only keep embeddings of words contained in conll vocab to reduce size of embeddings output file
### def export_reduced_embeddings
- loads specified embeddings and reduces dimensions with pca
- saves that to file 