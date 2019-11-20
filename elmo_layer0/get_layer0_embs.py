from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', required=True, help="Path to elmo weights file (.hdf5)")
parser.add_argument('-o', '--options', required=True, help="Path to elmo options file (.json)")
parser.add_argument('-v', '--vocab', '--vocabulary', required=True, help="Path to vocabulary file (one word per line).")
parser.add_argument('--cpu', action='store_true', help="Use cpu instead of gpu.")
args = parser.parse_args()

options_file = args.options
weight_file = args.weights
vocab_path = args.vocab
if args.cpu:
        cuda_rank = -1
    else:
        cuda_rank = 0


elmo = ElmoEmbedder(options_file, weight_file, cuda_rank)

vocab = []
embs = {}

with open(vocab_path, 'r') as vf:
    for line in vf:
        vocab.append(line.strip())
embcounter = 0
minibatches = [vocab[i:i+10] for i in range(0, len(vocab), 10)]

for i in range(0, len(minibatches), 10):
    batchtokens = minibatches[i:i+10]
    if (len(batchtokens), len(batchtokens[0]), len(batchtokens[-1])) != (10,10,10):
        print(len(batchtokens), len(batchtokens[0]), len(batchtokens[-1]))
    batchvectors = elmo.embed_batch(batchtokens)
    if (len(batchvectors), len(batchvectors[0][0]), len(batchvectors[-1][0])) != (10,10,10):
        print(len(batchvectors), len(batchvectors[0][0]), len(batchvectors[-1][0]))
    for s in range(len(batchvectors)):
        for w in range(len(batchvectors[s][0])):
            embs[batchtokens[s][w]] = batchvectors[s][0][w]
            embcounter += 1
            if embcounter%5000 == 0:
                print(embcounter, '/', len(vocab))
print(len(embs))
print(embcounter)
with open(vocab_path+'.emb.layer0', 'w') as outfile:
    buffer = str(len(embs))+" 1024\n"
    wordcount = 0
    for word in embs:
        buffer += word+' '+' '.join([str(v) for v in embs[word]])+'\n'
        wordcount += 1
        if wordcount >= 1000:
            wordcount = 0
            outfile.write(buffer)
            buffer = ""
    outfile.write(buffer)
