import torch
import torch.nn as nn
import models
import preprocessing
from collections import defaultdict
import time
import os



def Logger(elapsed_time, epoch, epochs, tr_loss, tr_acc, val_loss, val_acc):
    '''
        Logger function to track training progress

        Input: elapsed_time - the current elapsed training time
               epoch - current epoch
               epochs - total number of epochs
               tr_loss/val_loss - current training/validation loss
               tr_acc/val_acc - current training/validation accuracy
    '''

    tim = 'sec'
    if elapsed_time > 60 and elapsed_time <= 3600:
        elapsed_time /= 60
        tim = 'min'
    elif elapsed_time > 3600:
        elapsed_time /= 3600
        tim = 'hrs'
    elapsed_time = format(elapsed_time, '.2f')
    print(f'Elapsed time: {elapsed_time} {tim}  Epoch: {epoch}/{epochs}  ',
          f'Train Loss: {tr_loss:.4f}  Val Loss: {val_loss:.4f}  ',
          f'Train Acc: {tr_acc:.2f}%  Val Acc: {val_acc:.2f}%')


class EarlyStopping(object):
    '''
        Stops the training progress if the performance has not improved for
        a number of epochs to avoid overfitting
    '''
    def __init__(self, patience):
        super().__init__()
        self.best_loss = 1e5
        self.patience = patience
        self.nr_no_improved = 0

    def update(self, curr_loss):
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss
            self.nr_no_improved = 0
            return False
        else:
            self.nr_no_improved+=1
            if self.nr_no_improved >= self.patience:
                print(f'Early stopping! Model did not improve for last {self.nr_no_improved} epochs')
                return True


class rnn_params:
    rnn_type = 'lstm'
    emb_dim = 64
    rnn_size = 128
    dropout = 0.5
    lr = 1e-3
    batch_size = 64
    n_epochs = 10
    decay = 1e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patience = 5

def train_rnn(save_path = None):
    '''
        Training function for the rnn model that trains and validates the models performance
    '''

    dataloaders, vocab_size, n_classes = preprocessing.preprocess(rnn_params.batch_size)
    train_loader, val_loader = dataloaders
    model = models.RNNModel(rnn_type=rnn_params.rnn_type, voc_size=vocab_size,
                            emb_dim=rnn_params.emb_dim, rnn_size=rnn_params.rnn_size,
                            n_classes=n_classes)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=rnn_params.lr, weight_decay=rnn_params.decay)
    model.to(rnn_params.device)

    history = defaultdict(list)
    init_training_time = time.time()
    early_stopping = EarlyStopping(patience=rnn_params.patience)
    for epoch in range(1, rnn_params.n_epochs):
        model.train()
        n_correct, n_instances, total_loss = 0,0,0
        for inputs, labels in train_loader:
            inputs = inputs.to(rnn_params.device)
            labels = labels.to(rnn_params.device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            total_loss+=loss.item()
            n_instances+=labels.shape[0]
            predictions = outputs.argmax(dim=1)
            n_correct += (predictions == labels).sum().item()

            optimizer.zero_grad()
            loss_fn.backward()
            optimizer.step()
        epoch_loss = total_loss / (len(train_loader))
        epoch_acc = n_correct / n_instances

        n_correct_val, n_instances_val, total_loss_val = 0, 0, 0
        model.eval()
        for val_inp, val_lab in val_loader:
            val_inp = val_inp.to(rnn_params.device)
            val_lab = val_lab.to(rnn_params.device)

            val_out = model(val_inp)
            val_loss = loss_fn(val_out, val_lab)

            total_loss_val += val_loss.item()
            n_instances_val += val_lab.shape[0]
            val_preds = val_out.argmax(dim=1)
            n_correct_val += (val_preds == val_lab).sum().item()

        epoch_val_loss = total_loss_val / len(val_loader)
        epoch_val_acc = n_correct_val / n_instances_val

        curr_time = time.time()
        Logger(curr_time-init_training_time, epoch, rnn_params.n_epochs, epoch_loss,
               epoch_acc, epoch_val_loss, epoch_val_acc)

        history['training loss'].append(epoch_loss)
        history['training acc'].append(epoch_acc)
        history['validation loss'].append(epoch_val_loss)
        history['validation acc'].append(epoch_val_acc)

        early_stop_check = early_stopping.update(epoch_val_loss)
        if early_stop_check:
            models.ModelUtils.save_model(save_path=save_path, model=model)

    if save_path:
        models.ModelUtils.save_model(save_path=save_path, model=model)


    return history





