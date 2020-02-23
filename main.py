import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy

import argparse
import time


from models import *

import random
import numpy as np
import matplotlib.pyplot as plt


def evaluate(net, loader, criterion):
    total_loss = 0.0
    total_acc = 0.0
    total_epoch = 0
    for i, a in enumerate(loader):
        in1, in2 = a.text
        labels = a.label
        outputs = net(in1, in2)
        loss = criterion(outputs.squeeze(), labels.float())
        corr = (outputs > 0.5).squeeze().long() == labels
        total_acc += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)

    acc = float(total_acc) / total_epoch
    loss = float(total_loss) / (i + 1)
    return acc, loss


def main(args):
    ######
    # 3.2 Processing of the data
    # the code below assumes you have processed and split the data into
    # the three files, train.tsv, validation.tsv and test.tsv
    # and those files reside in the folder named "data".
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed = seed
    np.random.seed = seed
    torch.backends.cudnn.deterministic = True
    spacy_en = spacy.load('en')
    ######

    # 3.2.1
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    # 3.2.2
    train_data, val_data, test_data = data.TabularDataset.splits(
            path='data/', train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])


    # 3.2.3
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    # 3.2.4
    TEXT.build_vocab(train_data, val_data, test_data)

    # 4.1
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=args.emb_dim))
    vocab = TEXT.vocab

    print("Shape of Vocab:", TEXT.vocab.vectors.shape)



    ######

    # 5 Training and Evaluation

    if args.model == 'cnn':
        net = CNN(args.emb_dim, vocab, args.num_filt)
    elif args.model == 'baseline':
        net = Baseline(args.emb_dim, vocab)
    else:
        net = RNN(args.emb_dim, vocab, args.rnn_hidden_dim)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    loss_fnc = nn.BCEWithLogitsLoss()

    train_acc = np.zeros(args.epochs)
    train_loss = np.zeros(args.epochs)
    val_acc = np.zeros(args.epochs)
    val_loss = np.zeros(args.epochs)

    a = time.time()

    for epoch in range(args.epochs):
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_epoch = 0

        for i, batch in enumerate(train_iter):
            batch_input, batch_input_length = batch.text
            labels = batch.label
            optimizer.zero_grad()
            outputs = net(batch_input, batch_input_length)
            loss = loss_fnc(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            corr = (outputs > 0.5).squeeze().long() == labels
            total_train_acc += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_acc[epoch] = float(total_train_acc) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i + 1)
        val_acc[epoch], val_loss[epoch] = evaluate(net, val_iter, loss_fnc)

        print("Epoch %s: train acc: %s, train loss: %s, val acc: %s, val loss: %s" % (epoch, train_acc[epoch], train_loss[epoch], val_acc[epoch], val_loss[epoch]))

    b = time.time()
    train_time = b - a
    torch.save(net, 'model_%s.pt' % args.model)

    test_acc, test_loss = evaluate(net, test_iter, loss_fnc)
    print('test acc: %s, test loss: %s' % (test_acc, test_loss))
    print('model %s' % args.model)
    print('training takes: %s' % train_time)


    plt.figure()
    plt.title("Training Loss and Accuracy vs. Number of Epochs")
    plt.plot(np.array(np.arange(len(train_loss))), train_loss, color='orange', label='training loss')
    plt.plot(np.array(np.arange(len(train_acc))), train_acc, color='blue', label='training accuracy')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss and Accuracy")
    plt.legend()
    plt.show()


    plt.figure()
    plt.title("Training and Validation Loss vs. Number of Epochs")
    plt.plot(np.array(np.arange(len(train_loss))), train_loss, color='orange', label='training loss')
    plt.plot(np.array(np.arange(len(val_loss))), val_loss, color='blue', label='validation loss')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Training and Validation Accuracy vs. Number of Epochs")
    plt.plot(np.array(np.arange(len(train_acc))), train_acc, color='orange', label='training accuracy')
    plt.plot(np.array(np.arange(len(val_acc))), val_acc, color='blue', label='validation accuracy')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracies")
    plt.legend()
    plt.show()


    ######


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default='rnn',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)
