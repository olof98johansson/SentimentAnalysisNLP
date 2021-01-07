import data_cleaning
import twint_scraping
import os

def collect_dataset(paths, keywords, nr_of_tweets, keys_to_remove, hashtags_to_remove):

    root, ext = os.path.splitext(paths)
    json_path = root+'.json'
    csv_path = root+'.csv'

    twint_scraping.collect_tweets(keywords=keywords, nr_tweets=nr_of_tweets, output_file=json_path)
    dataset_list = data_cleaning.datacleaning(path=json_path, keys_to_remove=keys_to_remove,
                                              hashtags_to_remove=hashtags_to_remove, save_path=csv_path)

    return dataset_list



