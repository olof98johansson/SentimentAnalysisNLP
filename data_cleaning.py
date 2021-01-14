
import json
import csv
import re
import pandas as pd

def load_json(path):
    '''
    Loads collected data in json format, checks it and then converts to csv format

    Input: path - path and file name to the collected json data (type: string)

    Output: keys - list of features/keys of the dataframe (type: list of strings)
            df_list - list containing all the dataframes from the json data (type: list of dataframes)
    '''
    if not path.endswith('.json'):
        print('File path not JSON file...')
        return None

    with open(path, 'r', encoding='utf8') as handle:
        df_list = [json.loads(line) for line in handle]

    nr_keys = [len(df_list[i].keys()) for i in range(len(df_list))]
    if not all(k == nr_keys[0] for k in nr_keys):
        print('Some features missing, review the data!')
        return None

    else:
        keys = df_list[0].keys()
        return keys, df_list


def combine_and_label(paths, labels, train=True):
    '''
    Combining multiple collections of data files and adds corresponding label (i.e depressive or non-depressive).
    List of labels in correct order with respect to the paths order must be specified manually

    Input: paths - list containing all the paths to the json files (type: list of strings)
           labels - list containing all the labels to the corresponding json files (type: list of strings)

    Output: df_list - list of all the combined dataframes from the json data (type: list of dataframes)
    '''

    if not type(paths)==type(list()):
        print('"paths" argument is not of type list! Please pass list of the paths to the collected data to be combined!')
        return None
    if train:
        if not len(paths) == len(labels):
            print(f'Number of datafile paths of {len(paths)} is not the same as number of labels of {len(labels)}!')
            return None

    df_list = []
    for idx, path in enumerate(paths):
        try:
            curr_keys, curr_df_list = load_json(path)
        except Exception as e:
            print(f'Unable to load data from path "{path}", check path name and file!')
            print(f'Exception:\n{e}')
            return None
        for df in curr_df_list:
            if train:
                df['label'] = labels[idx]
            df_list.append(df)

    return df_list



def datacleaning(paths, labels, hashtags_to_remove = [], save_path=None, train=True):
    '''
    Cleans the data based on unwanted hashtags, duplication of tweets occured due
    to sharing of keywords, removal of mentions, urls, non-english alphabetic tokens
    and empty tweets obtained after cleaning

    Input: paths - list containing all the paths to the json files (type: list of strings)
           labels - list containing all the labels to the corresponding json files (type: list of strings)
           hashtags_to_remove - list containing hashtags wished to be removed (type: list of strings)
           save_path - path and file name to were to save the cleaned dataset (type: string or None)
           train - specify if it is training mode or not, i.e if to use labels or not (type: boolean)

    Output: dataset_doc - list of all the text documents and corresponding labels if train (type: list of strings)
            keys - list of features/keys of the dataframe (type: list of strings)

    '''
    if len(labels) > 0:
        train = True

    df_list = combine_and_label(paths, labels, train=train)

    # Remove tweets with specific hashtags
    nr_removed_tweets = 0
    for idx, df in enumerate(df_list):
        hashtags = df.copy()['hashtags']
        if any([h in hashtags_to_remove for h in hashtags]):
            df_list.pop(idx)
            print(f'Tweet nr {idx} removed!')
            nr_removed_tweets += 1

    print(f'Removed total of {nr_removed_tweets} tweets')

    # Removes duplicate of tweets
    unique_ids = {}
    for idx, df in enumerate(df_list):
        tweet_id = df.copy()['id']
        if not tweet_id in unique_ids:
            unique_ids[str(tweet_id)] = 1
        else:
            print('Found douplicate of tweet id, removing the duplicate!')
            df_list.pop(idx)


    # Cleaning the tweet texts
    for idx, df in enumerate(df_list):
        tweet = df.copy()['tweet']
        # Removing URLs
        tweet = re.sub(r"http\S+", " ", tweet)
        tweet = re.sub(r"\S+\.com\S", " ", tweet)

        # Remove mentions
        tweet = re.sub(r'\@\w+', ' ', tweet)

        # Remove non-alphabetic tokens
        tweet = re.sub('[^A-Za-z]', ' ', tweet.lower())

        # Remove double spacings
        tweet = re.sub(' +', ' ', tweet)

        # Remove from dataset if tweet empty after cleaning
        if tweet == 0:
            df_list.pop(idx)
        else:
            df['tweet'] = tweet

    print('Successfully cleaned data!')


    # Saving list of tweet dicts to csv format

    if save_path:
        print(f'Saving data...')
        if not save_path.endswith('.csv'):
            print('Save path is missing .csv format extension!')
            save_path = save_path + '.csv'
        try:
            with open(save_path, 'w', encoding='utf8', newline='') as output_file:
                csv_file = csv.DictWriter(output_file,
                                          fieldnames=df_list[0].keys(),
                                          )

                csv_file.writeheader()
                csv_file.writerows(df_list)
                print(f'Data succesfully saved to "{save_path}"')

        except Exception as e:
            print(f'Unable to save data to "{save_path}", check the path and data!')
            print(f'Exception:\n{e}')

    dataset_docs = [df['tweet'] for df in df_list]
    keys = df_list[0].keys()
    if train:
        dataset_labels = [df['label'] for df in df_list]
        return [dataset_docs, dataset_labels], keys
    else:
        return dataset_docs, keys
