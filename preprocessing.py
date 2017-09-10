PAD_TOKEN = 0
EOM_TOKEN = 1

def load_vocab(filename, vocab_offset):
  lines = open(filename).readlines()
  vocab = {
    word.strip(): i + vocab_offset
    for i, word in enumerate(lines)
  }
  inv = {
    v: k
    for k, v in vocab.items()
  }
  return (vocab, inv)


def load_glove_embeddings(vocab, embedding_filename):
  lines = open(filename).readlines()
  embeddings = {}
  for line in lines:
    word = line.split()[0]
    embedding = list(map(float, line.split()[1:]))
    if word in vocab:
      embeddings[vocab[word]] = embedding
  
  return embeddings

def read_messages(datafile):
  return ['im gay', 'same']

def tokenize_messages(msgs):
  sentences = []
  for msg in msgs:
    sentence = []
    import re
    for word in re.findall(r"[\w]+|[^\s\w]", msg):
      if word in vocab:
        sentence.append(vocab[word])
      else:
        # TODO ???
    sentences.append(sentence)
  return sentences
