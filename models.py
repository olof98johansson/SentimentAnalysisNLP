import torch
import torch.nn as nn
import preprocessing
import os
import numpy as np


class ModelUtils:
    def save_model(save_path, model):
        root, ext = os.path.splitext(save_path)
        if not ext:
            save_path = root + '.pth'
        try:
            torch.save(model.state_dict(), save_path)
            print(f'Successfully saved to model to "{save_path}"!')
        except Exception as e:
            print(f'Unable to save model, check save path!')
            print(f'Exception:\n{e}')

    def load_model(load_path, model):
        try:
            model.load_state_dict(torch.load(load_path))
            print(f'Successfully loaded the model from path "{load_path}"')

        except Exception as e:
            print(f'Unable to load the weights, check if different model or incorrect path!')

class RNNModel(nn.Module):

    def __init__(self, rnn_type, nr_layers, voc_size, emb_dim, rnn_size, dropout, n_classes):
        super().__init__()
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.nr_layers = nr_layers
        self.embedding = nn.Embedding(voc_size, emb_dim)

        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=emb_dim, hidden_size=rnn_size, dropout=dropout if nr_layers > 1 else 0,
                              bidirectional=False, num_layers=nr_layers, batch_first=True)

        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=rnn_size, dropout=dropout if nr_layers > 1 else 0,
                               bidirectional=False, num_layers=nr_layers, batch_first=True)

        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_size, dropout=dropout if nr_layers > 1 else 0,
                              bidirectional=False, num_layers=nr_layers, batch_first=True)

        else:
            print('Invalid or no choice for RNN type, please choose one of "rnn", "lstm" or "gru"')


        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=rnn_size, out_features=n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, hidden):
        self.batch_size = X.size(0)
        embedded = self.embedding(X)


        if self.rnn_type == 'rnn' or self.rnn_type == 'gru':
            rnn_out, final_state = self.rnn(embedded, hidden)

        elif self.rnn_type == 'lstm':
            rnn_out, hidden = self.rnn(embedded, hidden)

        else:
            print(f'Invalid rnn type! Rebuild the model with a correct rnn type!')
            return None

        rnn_out = rnn_out.contiguous().view(-1, self.rnn_size)
        drop = self.dropout(rnn_out)
        out = self.linear(drop)
        out = self.sigmoid(out)
        # reshape such batch size is first and get labels of last batch
        out = out.view(self.batch_size, -1)
        out = out[:, -1]

        return out, hidden

    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.nr_layers, batch_size, self.rnn_size)).to(device)
        c0 = torch.zeros((self.nr_layers, batch_size, self.rnn_size)).to(device)
        hidden = (h0, c0)
        return hidden



