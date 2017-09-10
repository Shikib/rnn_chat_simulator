import torch
import torch.optim
import torch.nn as nn

import models
import train

embedding_source = 'data/glove.6B.100d.txt'
vocab_source = 'data/vocabulary.txt'
hidden_size = 300
vocab_offset = 10 # Amount of space to reserve for things like EOS, pad symbol, ...

batch_size = 50
grad_clip = 50
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 50000

vocab, inversevocab = preprocessing.load_vocab(vocab_source, vocab_offset=vocab_offset)
embeddings = preprocessing.load_glove_embeddings(vocab, embedding_source)

vocab_size = embeddings[max[
input_size = len(embeddings[vocab_offset])

encoder = models.Encoder(
  input_size=input_size,
  hidden_size=hidden_size,
  vocab_size=vocab_size
  embedding_dict=embeddings,
  num_layers=1,
  dropout=0,
  rnn_type='gru')

decoder = models.Decoder(
  input_size=input_size,
  hidden_size=hidden_size,
  vocab_size=vocab_size,
  num_layers=1,
  dropout=0,
  rnn_type='gru')

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

epoch = 0
while epoch < n_epochs:
    epoch += 1
    
    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = data.random_batch(data, batch_size)

    # Run the train function
    loss, ec, dc = train(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion
    )

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    eca += ec
    dca += dc
    
    job.record(epoch, loss)

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)
        
    if epoch % evaluate_every == 0:
        evaluate_randomly()

    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
        
        # TODO: Running average helper
        ecs.append(eca / plot_every)
        dcs.append(dca / plot_every)
        ecs_win = 'encoder grad (%s)' % hostname
        dcs_win = 'decoder grad (%s)' % hostname
        vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
        vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
        eca = 0
        dca = 0

# train.train(..., grad_clip=grad_clip)
# predict.predict()
# sentence = [inversevocab[x] for x in prediction]
# ???
# meme
