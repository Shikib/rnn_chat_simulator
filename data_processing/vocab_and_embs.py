import csv

from collections import *
from gensim.models import Word2Vec

rows = [e for e in csv.reader(open("messages_dump.tsv"), delimiter="\t")]

vocab = Counter(" ".join([e[1] for e in rows]).split())
vocab_size = 10000

import pdb; pdb.set_trace()
vocab = ["<UNK>"] + [e[0] for e in vocab.most_common(vocab_size)]
vocab_set = set(vocab)

open("vocab.wl", "w+").writelines([e + "\n" for e in vocab])

split_sents = [ [w for w in e[1].split() if w in vocab_set] for e in rows ]
w2v = Word2Vec(split_sents, min_count=0)

words = w2v.wv.index2word
embs = w2v.wv.syn0

open("w2v.emb", "w+").writelines([
  words[i] + " " + " ".join([str(e) for e in embs[i]]) + "\n"
  for i in range(len(words))
])
