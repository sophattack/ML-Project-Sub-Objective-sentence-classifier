import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):

    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        #x = [sentence length, batch size]
        embedded = self.embedding(x)

        average = embedded.mean(0) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze()

	# Note - using the BCEWithLogitsLoss loss function
        # performs the sigmoid function *as well* as well as
        # the binary cross entropy loss computation
        # (these are combined for numerical stability)

        return output


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters):
        super(CNN, self).__init__()

        ######

        # Section 5.0 YOUR CODE HERE
        self.embedding_layer = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)
        self.conv_layer1 = nn.Conv2d(1, n_filters, (2, embedding_dim))
        self.conv_layer2 = nn.Conv2d(1, n_filters, (4, embedding_dim))

        ######


    def forward(self, x, lengths=None):
        ######

        # Section 5.0 YOUR CODE HERE
        tmp = self.embedding_layer(x)
        tmp = tmp.permute(1, 0, 2)
        tmp = tmp.view(tmp.shape[0], 1, tmp.shape[1], tmp.shape[2])
        a1 = F.relu(self.conv_layer1(tmp))
        a2 = F.relu(self.conv_layer2(tmp))
        a1 = F.max_pool2d(a1, (a1.shape[2], 1))
        a2 = F.max_pool2d(a2, (a2.shape[2], 1))
        tmp = torch.cat([a1, a2], 1)
        tmp = torch.squeeze(tmp)
        tmp = self.fc(tmp).squeeze()
        return tmp

        ######


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()

        ######

        # Section 6.0 YOUR CODE HERE
        self.embedding_layer = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc = nn.Linear(embedding_dim, 1)
        self.hidden = nn.GRU(embedding_dim, hidden_dim)

        ######

    def forward(self, x, lengths=None):

        ######

        # Section 6.0 YOUR CODE HERE
        tmp = self.embedding_layer(x)
        tmp = nn.utils.rnn.pack_padded_sequence(tmp, lengths, enforce_sorted=False)
        tmp, h_n = self.hidden(tmp)
        h_n = torch.squeeze(h_n)
        tmp = self.fc(h_n).squeeze()
        return tmp

        ######
