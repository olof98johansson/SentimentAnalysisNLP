import models
import train
import preprocessing
import data_cleaning
import os
import torch
import twint_scraping
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


class Config:
    '''
    Configuration class to store and tune global variables
    '''
    test_set_keywords = []
    test_set_nr_of_tweets = [2000, 2000, 2000, 2000, 2000, 2000, 2000]

    # Locations spread out in UK to cover as wide geographical range as possible
    test_set_locations = ['london', 'liverpool', 'manchester',
                          'newcastle', 'edinburgh', 'glasgow', 'norwich']

    len_locations = len(test_set_locations)
    time_to = twint_scraping.get_weeks([2019, 9, 24], [2020, 3, 24]) # UK lockdown and 6 months back
    time_from = twint_scraping.get_weeks([2020, 3, 24], [2020, 9, 24]) # UK lockdown and 6 months forward
    test_set_time_spans = []
    for tt in time_to:
        test_set_time_spans.append(tt)
    for tf in time_from:
        test_set_time_spans.append(tf)
    len_timespan = len(test_set_time_spans)

    test_set_json_paths = []
    for t_idx in range(len_timespan):
        time_spec_path = []
        for l_idx in range(len_locations):
            time_spec_path.append(f'./forecast_data/testdata_{l_idx}_{t_idx}.json')

        test_set_json_paths.append(time_spec_path)

    test_set_csv_paths = [f'./forecast_data/all_loc_{t_idx}.csv' for t_idx in range(len_timespan)]

    path_to_weights = './weights/first_real_lstm.pth'


class TestDataset(Dataset):
    '''
    Basic class for creating dataset from the test input data
    '''
    def __init__(self, X):
        self.X = X

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)



def get_testdata(paths, save_path, timespans, collect_test_data = False):
    '''
    Builds vocabulary and encoder based on the training data and collects, clean and builds data loaders
    for the test data

    Input: paths - path to store the collected test data with json extension (type: list of strings)
           save_path - path to where to save the cleaned and final test dataset with csv
                       extension (type: list of strings)
           timespans - timespans of when the collected test tweets where tweeted (type: list of lists of strings)
           collect_test_data - specifying if to collect test data or not (type: boolean)

    Output: test_loader - data loader for the collected test data (type: DataLoader)
            encoder - encoder trained on the training labels (type: LabelEncoder)
            vocab_size - size of the vocabulary built from the training data (type: int)
            n_classes: number of classes/labels from the training data (type: int)
    '''
    roots, exts = [], []
    for path in paths:
        root, ext = os.path.splitext(path)
        roots.append(root)
        exts.append(ext)
    save_root, save_exts = os.path.splitext(save_path)
    json_paths = [root + '.json' for root in roots]
    csv_path = save_root + '.csv'

    rnn_params = train.rnn_params()
    _, vocab_size, n_classes = preprocessing.preprocess(rnn_params.batch_size, collect=False)
    encoder = preprocessing.config.encoder
    vocab = preprocessing.config.vocab
    if collect_test_data:
        for idx, json_path in enumerate(json_paths):
            twint_scraping.collect_tweets(nr_tweets=Config.test_set_nr_of_tweets[idx],
                                          output_file=json_path,
                                          near=Config.test_set_locations[idx],
                                          timespan=timespans)
    testdata, keys = data_cleaning.datacleaning(paths=json_paths, labels=[],
                                               hashtags_to_remove=[],
                                               save_path=csv_path, train=False)

    pad = vocab.get_pad_idx()
    max_len = max(len(x) for x in testdata)
    testdata = vocab.encode(testdata)
    testdata_padded = torch.as_tensor([x + [pad] * (max_len - len(x)) for x in testdata])
    test_dataset = TestDataset(testdata_padded)
    test_loader = DataLoader(test_dataset, batch_size=1)

    return test_loader, encoder, vocab_size, n_classes


def predict(testdata, path_to_weights, vocab_size, n_classes):
    '''
    Creates, loads and initiates the model and making predictions on the test data

    Input: testdata - data loader of the test data (type: DataLoader)
           path_to_weights - relative path and file name of the saved model weights with .pth extension (type:string)
           vocab_size - size of the vocabulary (type: int)
           n_classes - number of labels/classes that can be predicted (type: int)

    Output: preds_prob_list - list of all the probabilities of which the model predicted
                              the corresponding label (type: list of floats)
            preds_status_list - list of all the reencoded labels that were predicted (type: list of strings)
    '''
    rnn_params = train.rnn_params
    model = models.RNNModel(rnn_type=rnn_params.rnn_type, nr_layers=rnn_params.nr_layers,
                            voc_size=vocab_size, emb_dim=rnn_params.emb_dim, rnn_size=rnn_params.rnn_size,
                            dropout=rnn_params.dropout, n_classes=n_classes)
    models.ModelUtils.load_model(path_to_weights, model)
    model.to(rnn_params.device)
    batch_size = 1
    h = model.init_hidden(batch_size, device=rnn_params.device)
    model.zero_grad()

    preds_prob_list, preds_status_list = [], []
    for x_test in testdata:
        x_test = x_test.to(train.rnn_params.device)
        h = tuple([each.data for each in h])
        out, h = model(x_test, h)
        pred = torch.round(out.squeeze()).item()
        pred_status = "depressive" if pred < 0.5 else "non-depressive"
        prob = (1-pred) if pred_status == "depressive" else pred
        preds_status_list.append(pred_status)
        preds_prob_list.append(prob)

    return preds_prob_list, preds_status_list


def run_predictions(collect_test_data=False):
    '''
    Collect, preprocess and predicts the test data

    Input: collect_test_data - weither or not to collect test data (type: boolean)

    Output: status_results - all the predicted labels (type: dictionary of lists of strings)
            preds_results - all the predicted values, i.e the certainties of
                            the predictions (type: dictionary of lists of strings)
    '''
    status_results = {}
    preds_results = {}
    try:
        if collect_test_data:
            for idx, ind_paths in enumerate(Config.test_set_json_paths):
                testdata, encoder, vocab_size, n_classes = get_testdata(ind_paths,
                                                 Config.test_set_csv_paths[idx],
                                                 timespans=Config.test_set_time_spans[idx],
                                                 collect_test_data=True)


        else:
            for idx, ind_paths in enumerate(Config.test_set_json_paths):
                testdata, encoder, vocab_size, n_classes = get_testdata(ind_paths,
                                                 Config.test_set_csv_paths[idx],
                                                 timespans=Config.test_set_time_spans[idx],
                                                 collect_test_data=False)
    except Exception as e:
        print(f'Exception:\n{e}')
        print(f'Unable to get test data!')
        return None

    for idx, ind_paths in enumerate(Config.test_set_json_paths):
        preds_list, preds_status_list = predict(testdata, Config.path_to_weights,
                                                encoder, vocab_size, n_classes)
        status_results[f'timespan_{idx}'] = preds_status_list
        preds_results[f'timespan_{idx}'] = preds_list
    return status_results, preds_results






def plot_predictions(status_results, preds_results, save_name='./predictions_forecast.png'):
    '''
    Plot the predictions in time order, i.e a time-based forecast of the predictions

    Input: status_results - all the predicted labels (type: dictionary of lists of strings)
           preds_results - all the predicted values, i.e the certainties of
                           the predictions (type: dictionary of lists of strings)
           save_name - path and filename to where to save the forecasting plot
    '''
    timespans = list(status_results.keys())
    nr_depressive = [(np.array(status_results[timespans[t_idx]]) == 'depressive').sum() for t_idx in range(len(timespans))]
    percentage_dep = [((np.array(status_results[timespans[t_idx]]) == 'depressive').sum())/len(status_results[timespans[t_idx]]) for t_idx in range(len(timespans))]
    text_perc_dep = [format(percentage_dep[i]*100, '.2f') for i in range(len(percentage_dep))]
    ave_probs = [np.mean(np.array(preds_results[timespans[t_idx]])) for t_idx in range(len(timespans))]
    text_ave_probs = [format(ave_probs[i]*100, '.2f') for i in range(len(ave_probs))]

    fig = plt.figure(figsize=(20, 10))
    plt.bar(timespans, percentage_dep, color="#ff3399", width=0.6, alpha = 0.7)
    for i, p in enumerate(percentage_dep):
        plt.text(timespans[i], p / 2, f'{text_perc_dep[i]}%', verticalalignment='center', color='black',
                horizontalalignment='center', fontweight='bold', fontsize=14)
        plt.text(timespans[i], p+0.005, f'Average target prob: {text_ave_probs[i]}%', verticalalignment='center',
                 horizontalalignment='center', color='black', fontweight='bold', fontsize=10)
    plt.xlabel('Time period', fontsize=14)
    plt.ylabel('Percentage %', fontsize=14)
    plt.title('Percentage of depressive tweets', fontsize=18)
    if save_name:
        root, ext = os.path.splitext(save_name)
        save_name = root + '.png'
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()



def run():
    '''
    Predict function to run the prediction process after specifying parameters
    '''
    preprocessing.config.paths = ['./training_data/depressive1.json',
                                  './training_data/depressive2.json',
                                  './training_data/depressive3.json',
                                  './training_data/depressive4.json',
                                  './training_data/non-depressive1.json',
                                  './training_data/non-depressive2.json',
                                  './training_data/non-depressive3.json',
                                  './training_data/non-depressive4.json']

    preprocessing.config.labels = ['depressive', 'depressive', 'depressive', 'depressive',
                                   'not-depressive', 'not-depressive','not-depressive','not-depressive']

    status_results, preds_results = run_predictions(collect_test_data=True) # collect_test_data=False if already collected
    plot_predictions(status_results, preds_results)

run()














