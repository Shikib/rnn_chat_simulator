import constants
import csv
import re

def load_vocab(filename):
  """
  Load the vocabulary
  """
  lines = open(filename).readlines()
  w2i = {
    word.strip():i
    for i,word in enumerate(lines)
  }
  i2w = {
    v:k
    for k,v in w2i.items()
  }
  return w2i, i2w

def load_embeddings(embedding_filename, vocab):
  """
  Load embeddings.

  In file: word [embedding vector]
  In output: word -> embedding_vector
  """
  lines = open(embedding_filename).readlines()
  embeddings = {}
  for line in lines:
    word = line.split()[0]
    embedding = list(map(float, line.split()[1:]))
    if word in vocab:
      embeddings[vocab[word]] = embedding
  
  return embeddings

def load_messages(data_file):
  """
  Load messages from data file.
  """
  messages = []
  for row in csv.reader(open(data_file), delimiter="\t"):
    messages.append((row[0], row[1]))

  return messages

def numberize_messages(messages, vocab):
  """
  Encode the messages. 

  Assume vocab["<UNK>"] = 0
  Assume messages have been preprocessed/tokenized already.
  """
  numberized_messages = []
  for sender,msg in messages:
    numberized = [vocab.get(word, constants.UNK) for word in msg.split()]
    numberized_messages.append((sender, numberized))

  return numberized_messages
