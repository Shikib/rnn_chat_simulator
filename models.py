"""
File which contains model definitions.
"""

import torch.nn as nn
import torch.nn.init
import preprocessing

USE_CUDA = True

def load_embedding(embedding_dict):
  embedding = nn.Embedding(vocab_size, input_size, sparse=False, padding_idx=0)
  embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
  torch.nn.init.uniform(embedding_weights, a=-0.25, b=0.25)
  for k, v in embedding_dict.items():
    embedding_weights[k] = torch.FloatTensor(v)
  # TODO embeddings for special tokens (EOS, ...)
  del embedding.weight
  embedding.weight = nn.Parameter(embedding_weights)

  return embedding

class Attn(nn.Module):
  def __init__(self, method, hidden_size):
    super(Attn, self).__init__()
    
    self.method = method
    self.hidden_size = hidden_size
    
    if self.method == 'general':
      self.attn = nn.Linear(self.hidden_size, hidden_size)

    elif self.method == 'concat':
      self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
      self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

  def forward(self, hidden, encoder_outputs):
    max_len = encoder_outputs.size(0)
    this_batch_size = encoder_outputs.size(1)

    # Create variable to store attention energies
    attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

    if USE_CUDA:
      attn_energies = attn_energies.cuda()

    # For each batch of encoder outputs
    for b in range(this_batch_size):
      # Calculate energy for each encoder output
      for i in range(max_len):
        attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

    # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
    return F.softmax(attn_energies).unsqueeze(1)
  
  def score(self, hidden, encoder_output):
    
    if self.method == 'dot':
      energy = hidden.dot(encoder_output)
      return energy
    
    elif self.method == 'general':
      energy = self.attn(encoder_output)
      energy = hidden.dot(energy)
      return energy
    
    elif self.method == 'concat':
      energy = self.attn(torch.cat((hidden, encoder_output), 1))
      energy = self.v.dot(energy)
      return energy

class Encoder(nn.Module):
  def __init__(
    self,
    input_size,
    hidden_size,
    vocab_size,
    embedding_dict,
    num_layers=1,
    dropout=0,
    rnn_type='gru',
  ):
    """
    Initialize the model.
    """
    super(Encoder, self).__init__()

    self.vocab_size = vocab_size
    self.input_size = input_size
    self.hidden_size = hidden_size 
    self.num_layers = num_layers
    self.rnn_type = rnn_type

    self.embedding = load_embedding(embedding_dict)

    if rnn_type == 'gru':
      self.rnn = nn.GRU(
        input_size,
        self.hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False,
      )
    else:
      self.rnn = nn.LSTM(
        input_size,
        self.hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False,
      )

    self.init_weights()

  def forward(self, input_seqs, input_lengths):
    """
    Forward pass for the model.

    Input -> Embed -> RNN -> Output
    """
    embedded = self.embedding(input_seqs)
    packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
    outputs, hidden = self.gru(packed, None)

    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs
    return outputs, hidden

  def init_weights(self):
    """
    Initialize weights for RNN.
    """
    # Common initialization strategy
    torch.nn.init.orthogonal(self.rnn.weight_ih_l0)
    torch.nn.init.uniform(self.rnn.weight_hh_l0, a=-0.01, b=0.01)

  def init_hidden(self):
    hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
    if USE_CUDA: 
      hidden = hidden.cuda()

    return hidden

class Decoder(nn.Module):
  def __init__(
    self,
    input_size,
    hidden_size,
    vocab_size,
    embedding_dict,
    num_layers=1,
    dropout=0,
    rnn_type='gru',
  ):
    """
    Initialize the model.
    """
    super(Decoder, self).__init__()

    self.vocab_size = vocab_size
    self.input_size = input_size
    self.hidden_size = hidden_size 
    self.num_layers = num_layers
    self.rnn_type = rnn_type

    # Define layers
    self.embedding = load_embedding(embedding_dict)
    self.attn = Attn(hidden_size)

    if rnn_type == 'gru':
      self.rnn = nn.GRU(
        input_size,
        self.hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
      )
    else:
      self.rnn = nn.LSTM(
        input_size,
        self.hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
      )
    self.out = nn.Linear(hidden_size, output_size)

    self.init_weights()

  def forward(self, last_output, last_hidden, encoder_outputs):
    """
    Run RNN decoder on one timestep, to generate one output word.
    """
    # Embed last output word
    embedded_word = self.embedding(last_output).view(1, 1, -1)
    # TODO the notebook does a dropout here
     
    # Calculate attention weights based on last_hidden 
    attn_weights = self.attn(last_hidden[-1], encoder_outputs)

    # Apply attention weights to encoder outputs
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

    # TODO I added the next line because it was in the batching notebook but idk why it's here
    context = context.transpose(0, 1)

    # Concatenate last output word with attended context and run through RNN
    rnn_input = torch.cat((embedded_word, context), 2)
    output, hidden = self.rnn(rnn_input, last_hidden)

    # Run through softmax layer
    output = output.squeeze(0)
    output = F.log_softmax(self.out(torch.cat((output, context), 1)))

    # Return softmax output and RNN hidden state 
    return output, hidden

  def init_weights(self):
    """
    Initialize weights for RNN.
    """
    # Common initialization strategy
    init.orthogonal(self.rnn.weight_ih_l0)
    init.uniform(self.rnn.weight_hh_l0, a=-0.01, b=0.01)
