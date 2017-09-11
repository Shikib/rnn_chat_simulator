"""
File which contains model definitions.
"""

import constants
import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

def load_embedding(embedding_dict, vocab_size, input_size):
  """
  Load the embeddings. 
  """
  embedding = nn.Embedding(vocab_size, input_size, sparse=False, padding_idx=0)
  embedding_weights = torch.FloatTensor(vocab_size, input_size)
  torch.nn.init.uniform(embedding_weights, a=-0.25, b=0.25)
  for k,v in embedding_dict.items():
    embedding_weights[k] = torch.FloatTensor(v)

  del embedding.weight
  embedding.weight = nn.Parameter(embedding_weights)

  return embedding

class Attention(nn.Module):
  def __init__(self, method, hidden_size):
    """
    Initialize the attention mechanism.
    """
    super(Attention, self).__init__()
    
    self.method = method
    self.hidden_size = hidden_size
    
    # Two different types of attention based on Luong et al. (arXiv:1508.04025)
    if self.method == 'general':
      # score(h_t, h_s) = h_t^T W_a h_s
      # self.attn is W_a (hidden_size -> hidden_size mapping)
      self.attn = nn.Linear(self.hidden_size, self.hidden_size)
    elif self.method == 'concat':
      # score(h_t, h_s) = v_t^T W_a [ h_t ; h_s ]
      # self.attn is W_a (2*hidden_size -> hidden_size mapping)
      self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
      # self.v is v_t (hidden_size -> 1 mapping)
      self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

  def forward(self, hidden, encoder_outputs):
    """
    Given the previous decoder hidden state and the encoder hidden states, output
    the attention weights.
    """
    max_len = encoder_outputs.size(0)
    batch_size = encoder_outputs.size(1)

    # Attention energies.
    attn_energies = Variable(torch.zeros(batch_size, max_len)) # B x S

    if constants.USE_CUDA:
      attn_energies = attn_energies.cuda()


    import time; start = time.time()

    # Iterate over each batch and each encoder output to construct scores.
    for i in range(max_len):
      attn_energies[:,i] = self.score(hidden, encoder_outputs[i])

    print(time.time() - start)

    # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
    return F.softmax(attn_energies).unsqueeze(1)
  
  def score(self, hidden, encoder_output):
    """
    Determine the weight of the encoder output given the decoder hidden state.
    """
    # Three different types of attention based on Luong et al. (arXiv:1508.04025)
    if self.method == 'dot':
      energy = hidden.squeeze().dot(encoder_output.squeeze())
    elif self.method == 'general':
      energy = self.attn(encoder_output)
      energy = hidden.squeeze().dot(energy.squeeze())
    elif self.method == 'concat':
      energy = self.attn(torch.cat((hidden, encoder_output), 1))
      energy = self.v.matmul(energy.transpose(0, 1)).squeeze()

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

    self.embedding = load_embedding(embedding_dict, 
                                    self.vocab_size, 
                                    self.input_size)

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

    # Packed sequence
    packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
    outputs, hidden = self.rnn(packed, None)

    # Unpack sequence
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) 

    return outputs, hidden

  def init_weights(self):
    """
    Initialize weights for RNN.
    """
    # Common initialization strategy
    torch.nn.init.orthogonal(self.rnn.weight_ih_l0)
    torch.nn.init.uniform(self.rnn.weight_hh_l0, a=-0.01, b=0.01)

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
    self.embedding = load_embedding(embedding_dict, 
                                    self.vocab_size, 
                                    self.input_size)

    # TODO: emperiment with alternate attention methods
    # Concat is really slow for some reason.
    self.attn = Attention('concat', hidden_size)

    if rnn_type == 'gru':
      self.rnn = nn.GRU(
        self.input_size+self.hidden_size,
        self.hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False,
      )
    else:
      self.rnn = nn.LSTM(
        self.input_size+self.hidden_size,
        self.hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False,
      )

    self.out = nn.Linear(2*hidden_size, vocab_size)

    self.init_weights()

  def forward(self, last_output, last_hidden, encoder_outputs):
    """
    Run RNN decoder on one timestep, to generate one output word.
    """
    # Embed last output word
    embedded_word = self.embedding(last_output).view(1, -1, self.input_size)
     

    # Calculate attention weights based on last_hidden 
    attn_weights = self.attn(last_hidden, encoder_outputs)

    # Apply attention weights to encoder outputs
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
    context = context.transpose(0, 1)

    # Concatenate last output word with attended context and run through RNN
    rnn_input = torch.cat((embedded_word, context), 2)
    output, hidden = self.rnn(rnn_input, last_hidden.unsqueeze(0))

    # Run through softmax layer
    output = output.squeeze(0)
    output = F.log_softmax(self.out(torch.cat((output, context.squeeze(0)), 1)))

    # Return softmax output and RNN hidden state 
    return output, hidden.squeeze(0)

  def init_weights(self):
    """
    Initialize weights for RNN.
    """
    # Common initialization strategy
    init.orthogonal(self.rnn.weight_ih_l0)
    init.uniform(self.rnn.weight_hh_l0, a=-0.01, b=0.01)
