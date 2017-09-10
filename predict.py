"""
File for greedy decoding using the Attention Seq2Seq model.
"""
import preprocessing

USE_CUDA = True

def predict(encoder, decoder, sentence, max_length):
  """
  Given the input sentence and the maximum length of the output, 
  generate and return the output sentence.
  """
  input_variable = preprocessing.create_variable(sentence)
  input_length = input_variable.size()[0]

  encoder_hidden = encoder.init_hidden()
  encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

  decoder_input = Variable(torch.LongTensor([[data.start_token_index]])
  decoder_hidden = Variable(torch.zeros(1, decoder.hidden_size))

  if USE_CUDA:
    decoder_input = decoder_input.cuda()
    decoder_hidden = decoder_hidden.cuda()

  # TODO: does this really make sense?
  decoder_hidden = encoder_hidden

  # Run through the coder until we've hit the maximum length or an end token
  decoded_words = []
  for i in range(max_length):
    decoder_output, decoder_hidden = decoder(decoder_input, 
                                             decoder_hidden, 
                                             encoder_outputs)
    
    # Select top word
    _, top_word = decoder.output.data.topk(1)
    word_index = top_word[0][0]
    if word_index = data.end_token_index:
      decoded_words.append(data.end_token)
      break
    else:
      decoded_words.append(data.index2word[word_index])

    # Set selected word as being next input
    decoder_input = Variable(torch.LongTensor([[word_index]]))
    if USE_CUDA:
      decoder_input = decoder_input.cuda()

  return decoded_words
