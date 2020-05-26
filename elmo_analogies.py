from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
from scipy import spatial
import time
from math import floor
import heapq
import argparse
from annoy import AnnoyIndex
from multiprocessing import Pool
import sys


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', required=True, help="Path to elmo weights file (.hdf5)")
parser.add_argument('-o', '--options', required=True, help="Path to elmo options file (.json)")
parser.add_argument('-l', '--lang', '--language', required=True, help="Two letter language code.")
parser.add_argument('--trees', type=int, default=15, help="Number of trees in knn search.")
parser.add_argument('--searchk', type=int)
parser.add_argument('-k', '--k', type=int, default=10, help="Parameter k in kNN.")
parser.add_argument('-e', '--embeddings', required=True, help="Path to elmo layer0 embeddings file.")
parser.add_argument('-m', '--sem_neigh', type=int, default=200, help="Embed only n closest (with respect to layer0) options for each semantic example.")
parser.add_argument('-n', '--syn_neigh', type=int, default=50, help="Embed only n closest (with respect to layer0) options for each syntactic example.")
parser.add_argument('--lowercase', action="store_true")
parser.add_argument('--cpu', action="store_true", help="Use cpu instead of gpu to embed with ELMo")
args = parser.parse_args()

options_file = args.options
weight_file = args.weights
embeddings_file = args.embeddings
lc = args.lang
if args.cpu:
    cuda_rank = -1
else:
    cuda_rank = 0


elmo = ElmoEmbedder(options_file, weight_file, cuda_rank)
csd = spatial.distance.cosine


vocab = {}
dot = ["."]
allranks = {}
for rank in ["rank", "cslsrank", "acc1", "csls1", "acc5", "csls5", "acc10", "csls10"]:
    allranks[rank] = {}

counter = {}
datasize = 20000


def concat_layers(item, layers):
    newitem = []
    for wordindex in range(len(item[0])):
        word = []
        for l in layers:
            word += list(item[l][wordindex])
        newitem.append(word)
    return newitem
    
def sum_layers(item, layers):
    newitem = []
    for wordindex in range(len(item[0])):
        word = np.zeros(len(item[0][wordindex]))
        for l in layers:
            word += item[l][wordindex]
        newitem.append(word)
    return newitem
     
def knn(neighbours, vector, keyindex, k):
    dist = []
    for n in neighbours:
        d = [csd(n[layer][keyindex], vector[layer][keyindex]) for layer in [0,1,2]]
        dist.append((sum(d)/len(d), [n[l][keyindex] for l in [0,1,2]]))
    knearest = heapq.nsmallest(k, dist, key=lambda x: x[0])
    knearest = [kn[1] for kn in knearest]
    return knearest
    
def knn2(hood, item, layer, keyindex, k):
    tree = AnnoyIndex(1024)
    item = sum_layers(item, layer)
    tree.add_item(0, item[keyindex])
    for i in range(len(hood)):
        member = sum_layers(hood[i], layer)
        tree.add_item(i+1, member[keyindex])
    tree.build(args.trees)
    if args.searchk:
        knearest_indices = tree.get_nns_by_item(0, k, search_k=args.searchk)
    else:
        knearest_indices = tree.get_nns_by_item(0, k)
    knearest = [[hood[i-1][l][keyindex] for i in knearest_indices] for l in layer]
    return knearest

def build_tree(batchvectors, layer, keys):
    tree = AnnoyIndex(1024)
    leftside = batchvectors[0][layer][keys[1]] - batchvectors[0][layer][keys[0]] + batchvectors[0][layer][keys[2]]
    tree.add_item(0, leftside)
    for i in range(len(batchvectors)):
        tree.add_item(i+1,batchvectors[i][layer][keys[3]]) #rightsides
    tree.build(args.trees)
    return tree

def knn_from_tree(tree, item_index, k):
    if args.searchk:
        knearest_indices = tree.get_nns_by_item(item_index, k, search_k=args.searchk)
    else:
        knearest_indices = tree.get_nns_by_item(item_index, k)
    return knearest_indices
    

def csls(x,y,xneigh,yneigh):
    def mean_sim(z, neigh, K):
        meansim = sum(map(lambda zn: csd(z,zn), neigh))/K
        #return sum/K
        return meansim
    similarity = 2*csd(x,y) - mean_sim(x, xneigh, len(xneigh)) - mean_sim(y, yneigh, len(yneigh))
    return similarity

print("Reading 0th layer embeddings")
# READ 0th LAYER EMBEDDINGS (FOR VOCABULARY/NN)
with open(embeddings_file, 'r') as infile:
    embeddings0 = {}
    id_to_word = []
    word_to_id = {}
    zeroth_tree = AnnoyIndex(1024)
    infile.readline()
    idcount = 0
    for line in infile:
        line = line.split()
        word = line[0]
        id_to_word.append(word)
        word_to_id[word] = idcount
        try:
            vector = np.asarray([float(v) for v in line[1:]])
        except:
            continue
        if len(vector) != 1024:
            continue
        else:
            embeddings0[word] = vector
            zeroth_tree.add_item(idcount, vector)
            idcount += 1 
    zeroth_tree.build(args.trees)


# READ SUPPORT TEXT TO FORM SENTENCES TO EMBED
with open('support_text/'+lc+'.txt', 'r') as support_text:
    text1 = support_text.readline().split()
    text2 = support_text.readline().split()
    text3 = support_text.readline().split()
    text4 = support_text.readline().split()
keys = [len(text1), len(text1)+len(text2)+1, len(text1)+len(text2)+len(text3)+2, 
        len(text1)+len(text2)+len(text3)+len(text4)+3]
datacount = 0


# PROCESS ANALOGIES
print("Processing analogies.")
with open(lc+'-analogies.txt', 'r') as analogyfile, open('results.1.5.'+lc+'.txt', 'w') as outfile:
    category = ''
    time0 = time.time()
    for line in analogyfile:
        scores = {}
        cslsscores = {}
        if ':' in line:
            category = line.split()[1]
            for rank in allranks:
                allranks[rank][category] = np.array([0,0,0])
            counter[category] = 0
            print('Current results:')
            for rank in allranks:
                    for category in allranks[rank]:
                        print(category, rank, [round(allranks[rank][category][i]/counter[category],3) for i in [0,1,2] if counter[category]>0])
            print('Calculating', category)
            if 'gram' in category:
                neigh_size = args.syn_neigh
            else:
                neigh_size = args.sem_neigh
            continue
        else:
            if args.lowercase:
                line = line.lower()
            counter[category] += 1
            keywords = line.split()
            tokens = text1+[keywords[0]]+text2+[keywords[1]]+text3+[keywords[2]]+text4
            correct = keywords[3]
            leftside0 = embeddings0[keywords[1]] - embeddings0[keywords[0]] + embeddings0[keywords[2]]
            vocab_id = zeroth_tree.get_nns_by_vector(leftside0, neigh_size)
            vocab_id += zeroth_tree.get_nns_by_vector(embeddings0[correct], neigh_size)
            vocab_id = list(set(vocab_id))
            vocab = list(map(lambda x: id_to_word[x], vocab_id))
            batchtokens = [tokens+[pick]+dot for pick in vocab]
            batchvectors = elmo.embed_batch(batchtokens)
            candidates = []
            leftside_layer0 = batchvectors[0][0][keys[1]] - batchvectors[0][0][keys[0]] + batchvectors[0][0][keys[2]]
            leftside_layer1 = batchvectors[0][1][keys[1]] - batchvectors[0][1][keys[0]] + batchvectors[0][1][keys[2]]
            leftside_layer2 = batchvectors[0][2][keys[1]] - batchvectors[0][2][keys[0]] + batchvectors[0][2][keys[2]]
            leftside = [leftside_layer0, leftside_layer1, leftside_layer2]
           
            tree_layer0 = build_tree(batchvectors, 0, keys)
            tree_layer1 = build_tree(batchvectors, 1, keys)
            tree_layer2 = build_tree(batchvectors, 2, keys)
            neigh_tree = [tree_layer0, tree_layer1, tree_layer2]
            candidates = np.asarray([knn_from_tree(t, 0, 5*args.k) for t in neigh_tree])
            candidates = np.unique(candidates)
            lneigh_ind = [knn_from_tree(t, 0, args.k) for t in neigh_tree]
            lneigh = [[batchvectors[i][layer][keys[3]] for i in lneigh_ind[layer] if i<len(batchvectors)] for layer in [0,1,2]]
           
            def calc_distance(tree_index):
                distance = []
                cslsdistance = []
                for layer in [0,1,2]:
                    rightside = batchvectors[tree_index-1][layer][keys[3]]
                    distance.append(csd(leftside[layer], rightside))
                    rneigh_ind = knn_from_tree(neigh_tree[layer], tree_index, args.k)
                    rneigh = [batchvectors[i][layer][keys[3]] for i in rneigh_ind if i<len(batchvectors)]
                    cslsdistance.append(csls(leftside[layer], rightside, lneigh[layer], rneigh))
                return (vocab[tree_index-1], distance, cslsdistance)

            with Pool(8) as p:
                distances = p.map(calc_distance, candidates)
            for element in distances:
                scores[element[0]] = element[1]
                cslsscores[element[0]] = element[2]
            is_better = np.array([0,0,0])
            csls_better = np.array([0,0,0])
            if correct in scores:
                for pick in scores:
                    is_better += [scores[pick][i] <= scores[correct][i] for i in range(3)]
                    csls_better += [cslsscores[pick][i] <= cslsscores[correct][i] for i in range(3)]
            else:
                is_better += [len(vocab)+1]*3
                csls_better += [len(vocab)+1]*3
            allranks["rank"][category] += is_better
            allranks["cslsrank"][category] += csls_better
            allranks["acc1"][category] += [i <= 1 for i in is_better]
            allranks["csls1"][category] += [i <= 1 for i in csls_better]
            allranks["acc5"][category] += [i <= 5 for i in is_better]
            allranks["csls5"][category] += [i <= 5 for i in csls_better]
            allranks["acc10"][category] += [i <= 10 for i in is_better]
            allranks["csls10"][category] += [i <= 10 for i in csls_better]

            datacount += 1
            if datacount % 50 == 0:
                eta = round((time.time()-time0)*(datasize/datacount - 1))
                print(round(datacount/datasize, 3), 'elapsed:', round(time.time()-time0), 'ETA:', floor(eta/60), 'm', eta%60, 's')
            if datacount % 500 == 0:
                print('Intermediate results:')
                for rank in allranks:
                    print(category, rank, [round(allranks[rank][category][i]/counter[category],3) for i in [0,1,2]])

    for rank in allranks:
        for category in allranks[rank]:
            outfile.write(category+" "+rank+" ["+", ".join([str(round(allranks[rank][category][i]/counter[category],3)) for i in [0,1,2]])+"]\n")
            print(category, rank, [round(allranks[rank][category][i]/counter[category],3) for i in [0,1,2]])

