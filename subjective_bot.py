"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""

import torch
import numpy as np
import torchtext
from torchtext import data
import spacy
from decimal import *
from models import *


def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(text)]

TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
LABELS = data.Field(sequential=False, use_vocab=False)

train_data, val_data, test_data = data.TabularDataset.splits(
        path='data/', train='train.tsv',
        validation='validation.tsv', test='test.tsv', format='tsv',
        skip_header=True, fields=[('Text', TEXT), ('Label', LABELS)]
)

TEXT.build_vocab(train_data)
TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
vocab = TEXT.vocab
sentence = str(input("Enter a sentence:\n"))

while 1:
    baseline = torch.load('model_baseline.pt')
    rnn = torch.load('model_rnn.pt')
    cnn = torch.load('model_cnn.pt')

    tokens = tokenizer(sentence)

    token_ints = [vocab.stoi[tok] for tok in tokens]

    lengths = torch.Tensor([len(token_ints)])

    out_base = torch.sigmoid(baseline(torch.LongTensor(token_ints).view(-1, 1), lengths))
    out_rnn = torch.sigmoid(rnn(torch.LongTensor(token_ints).view(-1, 1), lengths))
    out_cnn = torch.sigmoid(cnn(torch.LongTensor(token_ints).view(-1, 1), lengths))

    if (out_base > 0.5).squeeze().item():
        msg_base = "subjective"
    else:
        msg_base = "objective"

    if (out_cnn > 0.5).squeeze().item():
        msg_cnn = "subjective"
    else:
        msg_cnn = "objective"

    if (out_rnn > 0.5).squeeze().item():
        msg_rnn = "subjective"
    else:
        msg_rnn = "objective"

    res_base = out_base.detach().numpy()
    res_cnn = out_cnn.detach().numpy()
    res_rnn = out_rnn.detach().numpy()

    print("\nModel baseline: {} ({})".format(msg_base, format(res_base, ".3f")))
    print("Model cnn: {} ({})".format(msg_cnn, format(res_cnn, ".3f")))
    print("Model rnn: {} ({})\n".format(msg_rnn, format(res_rnn, ".3f")))

    sentence = input("Enter a sentence:\n")
