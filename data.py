import constants
import random
import torch

from torch.nn import functional
from torch.autograd import Variable

def pad_seq(seq, max_length):
  """
  Pad sequence to maximum length.
  """
  newseq = seq.copy()
  newseq += [constants.UNK for i in range(max_length - len(seq))]
  return newseq

def create_batches(all_messages, maximum_input_length=512, maximum_output_length=256, user_filter=None):
  batches = []
  for i in range(100, len(all_messages)):
    if all_messages[i][0] != user_filter and user_filter is not None:
      continue
    if len(all_messages[i][1]) > maximum_output_length:
      continue

    total_length = 0
    context_length = 0

    while True:
      nextmsg = len(all_messages[i - context_length - 1][1])
      if total_length + nextmsg > maximum_input_length:
        break
      total_length += nextmsg
      context_length += 1

    if context_length == 0:
      continue

    batches.append((total_length, i, context_length))

    batch = batches[-1]
    input_seq = []
    for sender, msg in all_messages[batch[1] - batch[2] : batch[1]]:
      input_seq = input_seq + msg
    if len(input_seq) != batch[0]:
      import pdb; pdb.set_trace()
  batches.sort(key=lambda x: x[0])
  return batches

def random_batch(all_messages, batches, batch_size):
  """
  Given all messages, context length, batch size and an optional user filter: return
  a random batch.
  """
  input_seqs = []
  target_seqs = []

  # Choose random pairs
  start_index = random.randint(0, len(batches) - batch_size)
  for i in range(batch_size):
    batch = batches[start_index + i]
    input_seq = []
    for sender, msg in all_messages[batch[1] - batch[2] : batch[1]]:
      input_seq = input_seq + msg

    input_seqs.append(input_seq)
    if len(input_seq) != batch[0]:
        import pdb; pdb.set_trace()
    target_seqs.append(all_messages[batch[1]][1])
 
  # Zip into pairs, sort by length (descending), unzip
  seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
  input_seqs, target_seqs = zip(*seq_pairs)
  
  # For input and target sequences, get array of lengths and pad with 0s to max length
  input_lengths = [len(s) for s in input_seqs]
  input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
  target_lengths = [len(s) for s in target_seqs]
  target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

  # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
  input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
  target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
  
  if constants.USE_CUDA:
    input_var = input_var.cuda()
    target_var = target_var.cuda()
      
  return input_var, input_lengths, target_var, target_lengths

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = Variable(torch.LongTensor(length))
    if constants.USE_CUDA:
        length = length.cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss
