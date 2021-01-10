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
    test_set_keywords = []
    test_set_nr_of_tweets = [1000, 1000, 1000, 1000, 1000, 1000]
    test_set_locations = ['london', 'liverpool', 'manchester',
                          'newcastle', 'edinburgh', 'glasgow']

    test_set_json_paths = [['test11.json', 'test21.json', 'test31.json',
                            'test41.json', 'test51.json', 'test61.json'],
                           ['test12.json', 'test22.json', 'test32.json',
                            'test42.json', 'test52.json', 'test62.json'],
                           ['test13.json', 'test23.json', 'test33.json',
                           'test43.json', 'test53.json', 'test63.json']
                           ]
    test_set_csv_paths = ['testalltime1.csv', 'testalltime2.csv', 'testalltime3.csv']
    test_set_time_spans = [["2016-10-29 00:00:00", "2016-11-29 12:15:19"],
                           ["2017-10-29 00:00:00", "2017-11-29 12:15:19"],
                            ["2018-10-29 00:00:00", "2018-11-29 12:15:19"]]

    path_to_weights = './initial_test.pth'

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


def predict(testdata, path_to_weights, encoder, vocab_size, n_classes):
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
    status_results = {}
    preds_results = {}
    if collect_test_data:
        for idx, ind_paths in enumerate(Config.test_set_json_paths):
            testdata, encoder, vocab_size, n_classes = get_testdata(ind_paths,
                                             Config.test_set_csv_paths[idx],
                                             timespans=Config.test_set_time_spans[idx],
                                             collect_test_data=True)

            preds_list, preds_status_list = predict(testdata, Config.path_to_weights,
                                                    encoder, vocab_size, n_classes)
            status_results[f'timespan_{idx}'] = preds_status_list
            preds_results[f'timespan_{idx}'] = preds_list

    else:
        for idx, ind_paths in enumerate(Config.test_set_json_paths):
            testdata, encoder, vocab_size, n_classes = get_testdata(ind_paths,
                                             Config.test_set_csv_paths[idx],
                                             timespans=Config.test_set_time_spans[idx],
                                             collect_test_data=False)

            preds_list, preds_status_list = predict(testdata, Config.path_to_weights,
                                                    encoder, vocab_size, n_classes)
            status_results[f'timespan_{idx}'] = preds_status_list
            preds_results[f'timespan_{idx}'] = preds_list

    return status_results, preds_results


def plot_predictions(status_results, preds_results, save_name='./predictions_forecast.png'):
    timespans = list(status_results.keys())
    nr_depressive = [(np.array(status_results[timespans[t_idx]]) == 'depressive').sum() for t_idx in range(len(timespans))]
    percentage_dep = [((np.array(status_results[timespans[t_idx]]) == 'depressive').sum())/len(status_results[timespans[t_idx]]) for t_idx in range(len(timespans))]
    text_perc_dep = [format(percentage_dep[i]*100, '.2f') for i in range(len(percentage_dep))]
    ave_probs = [np.mean(np.array(preds_results[timespans[t_idx]])) for t_idx in range(len(timespans))]
    text_ave_probs = [format(ave_probs[i]*100, '.2f') for i in range(len(ave_probs))]

    fig = plt.figure(figsize=(20, 10))
    plt.bar(timespans, percentage_dep, color="#ff3399", width=0.6, alpha = 0.7)
    for i, p in enumerate(percentage_dep):
        plt.text(timespans[i], p / 2, f'{text_perc_dep[i]}%', color='black', fontweight='bold', fontsize=14)
        plt.text(timespans[i], p+0.005, f'Average target prob: {text_ave_probs[i]}%', color='black', fontweight='bold', fontsize=10)
    plt.xlabel('Time period', fontsize=14)
    plt.ylabel('Percentage %', fontsize=14)
    plt.title('Percentage of depressive tweets', fontsize=18)
    if save_name:
        root, ext = os.path.splitext(save_name)
        save_name = root + '.png'
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()



def run():
    preprocessing.config.paths = ['depressive1.json',
                                  'depressive2.json',
                                  'depressive3.json',
                                  'non-depressive1.json',
                                  'non-depressive2.json',
                                  'non-depressive3.json']

    preprocessing.config.labels = ['depressive', 'depressive', 'depressive',
                                   'not-depressive', 'not-depressive', 'not-depressive']
    status_results, preds_results = run_predictions(collect_test_data=False)
    plot_predictions(status_results, preds_results)

run()














