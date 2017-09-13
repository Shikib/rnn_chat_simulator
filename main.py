import torch
import torch.optim
import torch.nn as nn

import predict
import constants
import data
import models
import preprocessing
import train
import time
import math

# File parameters
embedding_file = 'data/w2v.emb'
vocab_file = 'data/vocab.txt'
messages_file = 'data/messages.tsv'

# Batch parameters
batch_size = 64
context_length = 2
user_filter = "Exiatus"

# Model parameters
decoder_learning_ratio = 5.0
epochs = 5000
grad_clip = 50
hidden_size = 200
learning_rate = 0.0001

# Load vocab/embeddings
w2i, i2w = preprocessing.load_vocab(vocab_file)
embeddings = preprocessing.load_embeddings(embedding_file, w2i)
vocab_size = len(w2i)
input_size = len(embeddings[1])

print("Loaded %d words" % vocab_size)

# Load/numberize messages
messages = preprocessing.load_messages(messages_file)
numberized_messages = preprocessing.numberize_messages(messages, w2i)

print("Loaded %d messages" % len(messages))

# Create encoder
encoder = models.Encoder(
  input_size=input_size,
  hidden_size=hidden_size,
  vocab_size=vocab_size,
  embedding_dict=embeddings,
  num_layers=1,
  dropout=0,
  rnn_type='gru',
)

print("Encoder online")

# Create decoder
decoder = models.Decoder(
  input_size=input_size,
  hidden_size=hidden_size,
  vocab_size=vocab_size,
  embedding_dict=embeddings,
  num_layers=1,
  dropout=0,
  rnn_type='gru',
)

print("Decoder online")

if constants.USE_CUDA:
  encoder = encoder.cuda()
  decoder = decoder.cuda()

print("Synchronized with graphics unit")

# Create Adam optimizers. Decoder has 5* the learning rate of the encoder.
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

print("Optimizers online")

# Logging parameters
start = time.time()
print_loss_total = 0
print_every = 10
eval_every = 25

batches = data.create_batches(numberized_messages, maximum_input_length=128, maximum_output_length=128, user_filter=user_filter)
print("%d batches created" % len(batches))

def predict_sentence(author, message, strip_tokens=False):
  msg = (author, "__som__ %s __eom__" % message)
  numberize = preprocessing.numberize_messages([msg], w2i)[0][1]
  words = predict.predict(numberize, encoder, decoder, 100)
  return unparse_sentence(words, strip_tokens)

def unparse_sentence(indices, strip_tokens=False):
  if not isinstance(indices, list):
    indices = [i.data[0] for i in indices]
  return ' '.join([i2w[i] for i in indices if i > constants.EOM or not strip_tokens])

# Training loop.
for epoch in range(epochs):
  # Get training data for this cycle
  input_batches, input_lengths, target_batches, target_lengths = \
    data.random_batch(numberized_messages, batches, batch_size)

  # Run the train function
  loss, results = train.train(
    input_batches, 
    input_lengths, 
    target_batches, 
    target_lengths,
    encoder, 
    decoder,
    encoder_optimizer, 
    decoder_optimizer, 
    criterion,
  )

  '''
  print("Input Batch size: {0}".format(str(input_batches.size())))
  print("Output Batch size: {0}".format(str(target_batches.size())))
  for i in range(input_batches.size()[1]):
    print("%s -> %s" % (unparse_sentence(input_batches[:, i]), unparse_sentence(results[:, i])))
  '''

  # Keep track of loss
  print_loss_total += loss

  def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)
  def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "%s (- %s)" % (as_minutes(s), as_minutes(rs))

  if epoch % print_every == 0:
    print_loss_avg = print_loss_total / print_every
    print_loss_total = 0
    print_summary = '%s (%d %d%%) %.4f' % (time_since(start, (epoch + 1) / epochs), epoch, (epoch + 1) / epochs * 100, print_loss_avg)
    print(print_summary)

  if epoch % eval_every == 0:
    msg = "yo dude wtf is going on with that guy"
    print("%s -> %s" % (msg, predict_sentence("me", msg)))
    msg = "wat u doin"
    print("%s -> %s" % (msg, predict_sentence("me", msg)))
    msg = "check out this cool video"
    print("%s -> %s" % (msg, predict_sentence("me", msg)))
    

import pdb; pdb.set_trace()
