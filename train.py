import torch
import torch.nn as nn
import models
import preprocessing
from collections import defaultdict
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from celluloid import Camera


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
            else:
                return False


class rnn_params:
    '''
    Configuration to store and tune RNN specific hyperparameters
    '''
    rnn_type = 'lstm'
    emb_dim = 64
    rnn_size = 64
    nr_layers = 1
    dropout = 0.5
    lr = 1e-3
    batch_size = 64
    n_epochs = 30
    decay = 1e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    patience = 5

def train_rnn(save_path = None, collect=True):
    '''
    Training function for the rnn model that trains and validates the models performance

    Input: save_path - path and file name to where to save the trained weights (type: string)
           collect - specify if to collect data or not (type: boolean)

    Output: history - history of the models training progression (type: defaultdict of lists)
            early_stop_check - if early stopping has been executed or not (type: boolean)
    '''
    dataloaders, vocab_size, n_classes = preprocessing.preprocess(rnn_params.batch_size, collect=collect)
    train_loader, val_loader = dataloaders
    model = models.RNNModel(rnn_type=rnn_params.rnn_type, nr_layers=rnn_params.nr_layers,
                            voc_size=vocab_size, emb_dim=rnn_params.emb_dim, rnn_size=rnn_params.rnn_size,
                            dropout=rnn_params.dropout, n_classes=n_classes)

    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    #loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=rnn_params.lr, weight_decay=rnn_params.decay)
    model.to(rnn_params.device)

    history = defaultdict(list)
    init_training_time = time.time()
    early_stopping = EarlyStopping(patience=rnn_params.patience)

    for epoch in range(1, rnn_params.n_epochs):
        model.train()
        h = model.init_hidden(rnn_params.batch_size, device=rnn_params.device)
        n_correct, n_instances, total_loss = 0,0,0
        for inputs, labels in train_loader:
            model.zero_grad()
            inputs = inputs.to(rnn_params.device)
            labels = labels.to(rnn_params.device)
            h = tuple([each.data for each in h])
            outputs, h = model(inputs, h)

            loss = loss_fn(outputs.squeeze(), labels.float())

            total_loss+=loss.item()
            n_instances+=labels.shape[0]
            predictions = torch.round(outputs.squeeze())
            n_correct += (torch.sum(predictions == labels.float())).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = total_loss / (len(train_loader))
        epoch_acc = n_correct / n_instances

        n_correct_val, n_instances_val, total_loss_val = 0, 0, 0
        model.eval()
        val_h = model.init_hidden(rnn_params.batch_size, device=rnn_params.device)
        for val_inp, val_lab in val_loader:
            val_inp = val_inp.to(rnn_params.device)
            val_lab = val_lab.to(rnn_params.device)

            val_h = tuple([each.data for each in val_h])

            val_out, val_h = model(val_inp, val_h)
            val_loss = loss_fn(val_out.squeeze(), val_lab.float())

            total_loss_val += val_loss.item()
            n_instances_val += val_lab.shape[0]
            val_preds = torch.round(val_out.squeeze())
            n_correct_val += (torch.sum(val_preds == val_lab.float())).item()

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
            return history, early_stop_check

    if save_path:
        root, ext = os.path.splitext(save_path)
        save_path = root + '.pth'
        models.ModelUtils.save_model(save_path=save_path, model=model)


    return history, early_stop_check



def show_progress(history, save_name = None):
    fig, axes = plt.subplots(1, 2, figsize=(21, 7))
    fig.suptitle('Training progression', fontsize=18)
    axes[0].plot(history['training loss'], linewidth=2, color='#99ccff', alpha=0.9, label='Training')
    axes[0].plot(history['validation loss'], linewidth=2, color='#cc99ff', alpha=0.9, label='Validation')
    axes[0].set_xlabel(xlabel='Epochs', fontsize=12)
    axes[0].set_ylabel(ylabel=r'$\mathcal{L}(\hat{y}, y)$', fontsize=12)
    axes[0].set_title(label='Losses', fontsize=14)

    axes[1].plot(history['training acc'], linewidth=2, color='#99ccff', alpha=0.9, label='Training')
    axes[1].plot(history['validation acc'], linewidth=2, color='#cc99ff', alpha=0.9, label='Validation')
    axes[1].set_xlabel(xlabel='Epochs', fontsize=12)
    axes[1].set_ylabel(ylabel=r'%', fontsize=12)
    axes[1].set_title(label='Accuracies', fontsize=14)

    axes[0].legend()
    axes[1].legend()
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def animate_progress(history, save_path, early_stop_check):
    root, ext = os.path.splitext(save_path)
    save_path = root + '.gif'

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    camera = Camera(fig)
    fig.suptitle('Training progression', fontsize=18)
    axes[0].set_xlabel(xlabel='Epochs', fontsize=12)
    axes[0].set_ylabel(ylabel=r'$\mathcal{L}(\hat{y}, y)$', fontsize=12)
    axes[0].set_title(label='Losses', fontsize=14)

    axes[1].set_xlabel(xlabel='Epochs', fontsize=12)
    axes[1].set_ylabel(ylabel=r'%', fontsize=12)
    axes[1].set_title(label='Accuracies', fontsize=14)

    epochs = np.arange(len(history['training loss']))

    for e in epochs:
        axes[0].plot(epochs[:e], history['training loss'][:e], linewidth=2, color='#99ccff')
        axes[0].plot(epochs[:e], history['validation loss'][:e], linewidth=2,  color='#cc99ff')

        axes[1].plot(epochs[:e], history['training acc'][:e], linewidth=2, color='#99ccff')
        axes[1].plot(epochs[:e], history['validation acc'][:e], linewidth=2, color='#cc99ff')
        axes[0].legend(['Training', 'Validation'])
        axes[1].legend(['Training', 'Validation'])
        camera.snap()


    for i in range(10):
        axes[0].plot(epochs, history['training loss'], linewidth=2, color='#99ccff')
        axes[0].plot(epochs, history['validation loss'], linewidth=2, color='#cc99ff')

        axes[1].plot(epochs, history['training acc'], linewidth=2, color='#99ccff')
        axes[1].plot(epochs, history['validation acc'], linewidth=2, color='#cc99ff')

        axes[0].legend(['Training', 'Validation'])
        axes[1].legend(['Training', 'Validation'])

        camera.snap()

    animation = camera.animate()
    animation.save(save_path, writer='imagemagick')








