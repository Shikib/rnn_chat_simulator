"""
File which contains model definitions.
"""

class Encoder(nn.Module):
  def __init__(
    self,
    input_size,
    hidden_size,
    vocab_size,
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

    self.embedding = nn.Embedding(vocab_size, input_size, sparse=False, padding_idx=0)

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

  def forward(self, inps):
    """
    Forward pass for the model.

    Input -> Embed -> RNN -> Output
    """
    embs = self.embedding(inps)
    outputs, hiddens = self.rnn(embs)
    return outputs, hiddens

  def init_weights(self):
    """
    Initialize weights for RNN/embedding.
    """
    # Common initialization strategy
    init.orthogonal(self.rnn.weight_ih_l0)
    init.uniform(self.rnn.weight_hh_l0, a=-0.01, b=0.01)

    # Pre-trained embeddings
    pretrained_embeddings = preprocessing.load_embeddings()
    embedding_weights = torch.FloatTensor(self.vocab_size, self.input_size)
    init.uniform(embedding_weights, a=-0.25, b=0.25)
    for k,v in pretrained_embeddings.items():
      embedding_weights[k] = torch.FloatTensor(v)
    embedding_weights[0] = torch.FloatTensor([0]*self.input_size)
    del self.embedding.weight
    self.embedding.weight = nn.Parameter(embedding_weights)


class Decoder(nn.Module):
  def __init__(
    self,
    input_size,
    hidden_size,
    vocab_size,
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
    self.embedding = nn.Embedding(vocab_size, input_size, sparse=False, padding_idx=0)
    self.attn = GeneralAttn(hidden_size)
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
     
    # Calculate attention weights based on last_hidden 
    attn_weights = self.attn(last_hidden[-1], encoder_outputs)

    # Apply attention weights to encoder outputs
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

    # Concatenate last output word with attended context and run through RNN
    rnn_input = torch.cat((embedded_word, context), 2)
    output, hidden = self.rnn(rnn_input, last_hidden)

    # Run through softmax layer
    output = output.squeeze(0)
    output = F.log_softmax(self.out(torch.cat((output, context), 1)))

    # Return softmax output and RNN hidden state 
    return output, hidden
