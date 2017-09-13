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
user_filter = None

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

if constants.USE_CUDA:
  encoder = encoder.cuda()
  decoder = decoder.cuda()

# Create Adam optimizers. Decoder has 5* the learning rate of the encoder.
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

# Logging parameters
start = time.time()
print_loss_total = 0
print_every = 5
eval_every = 50

# Training loop.
for epoch in range(epochs):
  # Get training data for this cycle
  input_batches, input_lengths, target_batches, target_lengths = \
    data.random_batch(numberized_messages, context_length, batch_size, user_filter=user_filter)

  # Run the train function
  loss = train.train(
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
    msg = ("me", "__som__ yo dude wtf is going on with that guy __eom__")
    numberize = preprocessing.numberize_messages([msg], w2i)[0][1]
    words = predict.predict(numberize, encoder, decoder, 100)
    sentence = ' '.join([i2w[i] for i in words])
    print("%s -> %s" % (msg, sentence))
    

import pdb; pdb.set_trace()
