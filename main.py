import torch
import torch.optim
import torch.nn as nn

import data
import models
import preprocessing
import train

# File parameters
embedding_file = 'data/w2v.emb'
vocab_file = 'data/vocab.txt'
messages_file = 'data/messages.tsv'

# Batch parameters
batch_size = 50
context_length = 3
user_filter = None

# Model parameters
decoder_learning_ratio = 5.0
epochs = 50000
grad_clip = 50
hidden_size = 300
learning_rate = 0.0001

# Load vocab/embeddings
w2i, i2w = preprocessing.load_vocab(vocab_file)
embeddings = preprocessing.load_embeddings(embedding_file, w2i)
vocab_size = len(w2i)
input_size = len(embeddings[1])

# Load/numberize messages
messages = preprocessing.load_messages(messages_file)
numberized_messages = preprocessing.numberize_messages(messages, w2i)

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

# Create Adam optimizers. Decoder has 5* the learning rate of the encoder.
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

# Logging parameters
print_every = 10
eval_every = 100

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

  if epoch % print_every == 0:
    print_loss_avg = print_loss_total/print_every
    print_loss_total = 0
    print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
    print(print_summary)

  if epoch % eval_every == 0:
    # TODO: add evaluation code (print out seq -> output)
    print("NOT IMPLEMENTED YET")

import pdb; pdb.set_trace()
