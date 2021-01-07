import torch
import torch.nn as nn
import preprocessing
import os


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

    def __init__(self, rnn_type, voc_size, emb_dim, rnn_size, dropout, n_classes):
        super().__init__()

        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(voc_size, emb_dim)

        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=emb_dim, hidden_size=rnn_size,
                              bidirectional=True, num_layers=1, batch_first=True)

        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=emb_dim, hidden_size=rnn_size,
                               bidirectional=True, num_layers=1, batch_first=True)

        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=emb_dim, hidden_size=rnn_size,
                              bidirectional=True, num_layers=1, batch_first=True)

        else:
            print('Invalid or no choice for RNN type, please choose one of "rnn", "lstm" or "gru"')


        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=2 * rnn_size, out_features=n_classes)

    def forward(self, X):
        embedded = self.embedding(X)

        if self.rnn_type == 'rnn' or self.rnn_type == 'gru':
            rnn_out, final_state = self.rnn(embedded)

        if self.rnn_type == 'lstm':
            rnn_out, (final_state, _) = self.rnn(embedded)

        top_forward = final_state[-2]
        top_backward = final_state[-1]
        top_both = torch.cat([top_forward, top_backward], dim=1)
        drop = self.dropout(top_both)
        out = self.linear(drop)
        return out


