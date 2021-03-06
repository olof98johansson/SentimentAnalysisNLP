import data_cleaning
import twint_scraping
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch




class config:
    '''
    Configuration class to store and tune global variables
    '''

    PAD = '___PAD___'
    UNKNOWN = '___UNKNOWN___'

    paths = ['./training_data/depressive1.json',
             './training_data/depressive2.json',
             './training_data/depressive3.json',
             './training_data/depressive4.json',
             './training_data/depressive5.json',
             './training_data/depressive6.json',
             './training_data/non-depressive1.json',
             './training_data/non-depressive2.json',
             './training_data/non-depressive3.json',
             './training_data/non-depressive4.json',
             './training_data/non-depressive5.json',
             './training_data/non-depressive6.json']

    labels = ['depressive', 'depressive', 'depressive', 'depressive', 'depressive', 'depressive',
              'not-depressive', 'not-depressive', 'not-depressive', 'not-depressive',
              'not-depressive', 'not-depressive']

    save_path = './training_data/all_training_data.csv'
    keywords = ['depressed', 'lonely', 'sad', 'depression', 'tired', 'anxious',
                'happy', 'joy', 'thankful', 'hope', 'hopeful', 'glad']
    nr_of_tweets = [5000, 5000, 5000, 5000, 5000, 5000,
                    5000, 5000, 5000, 5000, 5000, 5000]
    hashtags_to_remove = []
    encoder = None
    vocab = None
    vocab_size = 0
    n_classes = 0




def collect_dataset(paths, keywords, nr_of_tweets, hashtags_to_remove, collect=True):
    '''
    Collecting the dataset and cleans the data

    Input: paths - path to where to save the collected tweets (type: list of strings)
           keywords - keywords to be used for collecting tweets (type: list of strings)
           nr_of_tweets - number of tweets to be collected for each collecting process (type: list of ints)
           collect - specifying if to collect tweets or not (type: boolean)

    Output: dataset - cleaned dataset of the tweet texts and their labels (type: list if lists)
    '''
    roots, exts = [], []
    for path in paths:
        root, ext = os.path.splitext(path)
        roots.append(root)
        exts.append(ext)
    #roots, exts = [os.path.splitext(path) for path in paths]
    save_root, save_exts = os.path.splitext(config.save_path)
    json_paths = [root+'.json' for root in roots]
    csv_path = save_root+'.csv'
    if collect:
        for idx, json_path in enumerate(json_paths):
            twint_scraping.collect_tweets(keywords=keywords[idx], nr_tweets=nr_of_tweets[idx], output_file=json_path)

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


def preprocess(batch_size=64, collect=True):
    '''
    Function for preprocessing the data which splits the data into train/val, builds the vocabulary, fits
    the label encoder and creates the dataloaders for the train and validation set

    Input: batch_size - batch size to be used in the data loaders (type: int)
           collect - specifying if to collect data or not (type: boolean)

    Output: dataloaders - the created data loaders for training and validation set (type: list of data loaders)
            vocab_size - size of the built vocabulary (type: int)
            n_classes - number of classes/ladels in the dataset
    '''
    data, keys = collect_dataset(paths=config.paths, keywords=config.keywords,
                           nr_of_tweets=config.nr_of_tweets,
                           hashtags_to_remove=config.hashtags_to_remove,
                                 collect=collect)
    X, Y = data
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=1)

    vocab = Vocab()
    vocab.build_vocab(x_train)
    config.vocab = vocab

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
                              shuffle=True, collate_fn=batcher, drop_last=True)
    val_dataset = DocumentDataset(vocab.encode(x_val), encoder.transform(y_val))
    val_loader = DataLoader(val_dataset, batch_size,
                            shuffle=True, collate_fn=batcher, drop_last=True)
    dataloaders = [train_loader, val_loader]

    return dataloaders, vocab_size, n_classes










