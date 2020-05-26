#!/bin/bash

# Usage: ./elmo_analogies.sh path/to/vocabulary.file path/to/analogy_dataset path/to/elmo_weights.hdf5 path/to/elmo_options.json 2_letter_language_code 

vocab_orig=$1
analogies=$2
weights=$3
options=$4
lang=$5
vocab=elmo_layer0/${lang}-vocab.txt

head -n 300000 ${vocab_orig} > $vocab
grep -v ':' $analogies | awk '{print $1"\n"$2}' | sort -u >> $vocab
python3 elmo_layer0/get_layer0_embs.py --weights $weights --options $options --vocab $vocab

layer0=${vocab}.emb.layer0
python3 elmo_analogies.py --weights $weights --options $options --lang $lang --embeddings $layer0 2> log.txt

