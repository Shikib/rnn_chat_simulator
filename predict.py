"""
File for greedy decoding using the Attention Seq2Seq model.
"""
import data
import preprocessing
import torch
import constants
import numpy as np
from torch.autograd import Variable

def predict(sequence, encoder, decoder, max_length):
  """
  Given the input sentence and the maximum length of the output, 
  generate and return the output sentence.
  """
  # Set train to False to eliminate dropout (not necessary right now but 
  # might as well).
  encoder.train(False)
  decoder.train(False)

  # Create input batch
  # import pdb; pdb.set_trace()
  input_batch = Variable(torch.LongTensor([sequence]), volatile=True).transpose(0, 1) 

  if constants.USE_CUDA:
    input_batch = input_batch.cuda()

  # Run through encoder
  encoder_outputs, encoder_hidden = encoder(input_batch, [len(sequence)])

  # Create decoder output
  decoder_input = Variable(torch.LongTensor([[constants.SOM]]))
  decoder_hidden = Variable(torch.zeros(1, decoder.hidden_size))

  if constants.USE_CUDA:
    decoder_input = decoder_input.cuda()
    decoder_hidden = decoder_hidden.cuda()

  # TODO: does this really make sense?
  decoder_hidden = encoder_hidden[encoder.num_layers-1]

  # Run through the coder until we've hit the maximum length or an end token
  decoded_words = []
  for i in range(max_length):
    decoder_output, decoder_hidden = decoder(decoder_input, 
                                             decoder_hidden, 
                                             encoder_outputs)
    
    # Select top word
    word_index = np.random.choice(decoder_output.size()[1], p=torch.nn.functional.softmax(decoder_output.data)[0].data.cpu().numpy())
    decoded_words.append(word_index)

    if word_index == constants.EOM:
      break

    # Set selected word as being next input
    decoder_input = Variable(torch.LongTensor([[word_index]]))
    if constants.USE_CUDA:
      decoder_input = decoder_input.cuda()

  # Set train to True to go back to training (not necessary right now but 
  # might as well).
  encoder.train(True)
  decoder.train(True)

  return decoded_words
