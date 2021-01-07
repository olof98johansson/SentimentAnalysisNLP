import data_cleaning
import twint_scraping
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn


class config:
    '''
        Configuration class to store and tune global variables
    '''

    PAD = '___PAD___'
    UNKNOWN = '___UNKNOWN___'

    paths = []
    labels = ['depressed', 'depressed', 'depressed', 'not-depressed', 'not-depressed'] #example only
    save_path = './training_data.csv'
    keywords = []
    nr_of_tweets = 100000 # example
    hashtags_to_remove = []



def collect_dataset(paths, keywords, nr_of_tweets, hashtags_to_remove):
    '''
        Collecting the dataset and cleans the data
    '''

    roots, exts = [os.path.splitext(path) for path in paths]
    save_root, save_exts = os.path.splitext(config.save_path)
    json_paths = [root+'.json' for root in roots]
    csv_path = save_root+'.csv'

    twint_scraping.collect_tweets(keywords=keywords, nr_tweets=nr_of_tweets, output_file=json_path)
    dataset, keys = data_cleaning.datacleaning(paths=json_paths, labels=config.labels, hashtags_to_remove=hashtags_to_remove,
                                                    save_path=csv_path)

    return dataset, keys


class DocumentDataset(Dataset):
    '''
        Basic class for creating dataset from the input and label data
    '''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


class DocumentBatcher:
    '''
        Process the batches to desired output by transform into torch tensors and pads uneven input text data
        to the same length
    '''
    def __init__(self, voc):
        self.pad = voc.get_pad_idx()

    def __call__(self, XY):
        max_len = max(len(x) for x, _ in XY)
        Xpadded = torch.as_tensor([x + [self.pad] * (max_len - len(x)) for x, _ in XY])
        Y = torch.as_tensor([y for _, y in XY])

        return Xpadded, Y

class Vocab:
    '''
        Encoding the documents
    '''

    def __init__(self):
        # Splitting the tweets into words as tokenizer
        self.tokenizer = lambda s: s.split()

    def build_vocab(self, docs):
        '''
            Building the vocabulary from the documents, i.e creating the
            word-to-encoding and encoding-to-word dicts

            Input: docs - list of all the lines in the corpus
        '''

        freqs = Counter(w for doc in docs for w in self.tokenizer(doc))
        freqs = sorted(((f, w) for w, f in freqs.items()), reverse=True)


        self.enc_to_word = [config.PAD, config.UNKNOWN] + [w for _, w in freqs]

        self.word_to_enc = {w: i for i, w in enumerate(self.enc_to_word)}

    def encode(self, docs):
        '''
            Encoding the documents

            Input: docs - list of all the lines in the corpus
        '''
        unkn_index = self.word_to_enc[config.UNKNOWN]
        return [[self.word_to_enc.get(w, unkn_index) for w in self.tokenizer(doc)] for doc in docs]

    def get_unknown_idx(self):
        return self.word_to_enc[config.UNKNOWN]

    def get_pad_idx(self):
        return self.word_to_enc[config.PAD]

    def __len__(self):
        return len(self.enc_to_word)


def preprocess(batch_size = 64):
    '''
        Function for preprocessing the data which splits the data into train/val, builds
        the vocabulary, fits the label encoder and creates the dataloaders
    '''
    data, keys = collect_dataset(paths=config.paths, keywords=config.keywords,
                           nr_of_tweets=config.nr_of_tweets,
                           hashtags_to_remove=config.hashtags_to_remove)
    X, Y = data
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)

    vocab = Vocab()
    vocab.build_vocab(x_train)

    encoder = LabelEncoder()
    encoder.fit(y_train)
    config.encoder = encoder

    vocab_size = len(vocab)
    n_classes = len(encoder.classes_)
    config.vocab_size = vocab_size
    config.n_classes = n_classes

    batcher = DocumentBatcher(vocab)
    train_dataset = DocumentDataset(vocab.encode(x_train),
                                    encoder.transform(y_train))
    train_loader = DataLoader(train_dataset, batch_size,
                              shuffle=True, collate_fn=batcher)
    val_dataset = DocumentDataset(vocab.encode(x_val), encoder.transform(y_val))
    val_loader = DataLoader(val_dataset, batch_size,
                            shuffle=True, collate_fn=batcher)
    dataloaders = [train_loader, val_loader]

    return dataloaders, vocab_size, n_classes










