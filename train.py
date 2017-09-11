import data
import torch

def train(
  input_batches,
  input_lengths,
  target_batches,
  target_lengths,
  encoder,
  decoder,
  encoder_optimizer,
  decoder_optimizer,
  criterion,
  grad_clip=5.0,
):
  """
  Training function
  """
  # Zero gradients of both optimizers
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  # Initialize loss
  loss = 0 

  # Run words through encoder
  encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)
  
  # Prepare input and output variables
  decoder_input = Variable(torch.LongTensor([constants.SOM] * batch_size))

  # TODO: This seems wrong?
  decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

  max_target_length = max(target_lengths)
  all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

  # Move new Variables to CUDA
  if USE_CUDA:
    decoder_input = decoder_input.cuda()
    all_decoder_outputs = all_decoder_outputs.cuda()

  # Run through decoder one time step at a time
  for t in range(max_target_length):
    decoder_output, decoder_hidden, decoder_attn = decoder(
      decoder_input, decoder_hidden, encoder_outputs
    )

    all_decoder_outputs[t] = decoder_output

    # Teacher force by setting the next input to be target for current timestep.
    decoder_input = target_batches[t] 

  # Loss calculation and backpropagation
  loss = data.masked_cross_entropy(
    all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
    target_batches.transpose(0, 1).contiguous(), # -> batch x seq
    target_lengths
  )
  loss.backward()
  
  # Clip gradient norms
  torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
  torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

  # Update parameters with optimizers
  encoder_optimizer.step()
  decoder_optimizer.step()
  
  return loss.data[0]
