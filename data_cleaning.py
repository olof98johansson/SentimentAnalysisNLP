'''
    Basic data cleaning script of checking the collected dataset and removing unwanted
    keys and features. Improvements of further cleaning functionality will be added
'''

import json
import csv

def load_json(path):
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
        print(f'Original key features:\n{df_list[0].keys()}')
        return df_list[0].keys(), df_list


def datacleaning(path, keys_to_remove=[], hashtags_to_remove = [], save_path=None):
    keys, df_list = load_json(path)

    # Remove unwanted key features
    for df in df_list:
        for d in df.copy():
            if d in df and d in keys_to_remove:
                df.pop(d)

    print(f'\nKey features after cleaning:\n{df_list[0].keys()}')

    # Remove tweets with specific hashtags
    nr_removed_tweets = 0
    for idx, df in enumerate(df_list):
        hashtags = df.copy()['hashtags']
        if any([h in hashtags_to_remove for h in hashtags]):
            df_list.pop(idx)
            print(f'Tweet nr {idx} removed!')
            nr_removed_tweets += 1

    print(f'Removed total of {nr_removed_tweets}')

    # Saving list of tweet dicts to csv format
    if save_path:
        if not save_path.endswith('.csv'):
            print('Save path is missing .csv format extension!')
            save_path = save_path + '.csv'

        with open(save_path, 'w', encoding='utf8', newline='') as output_file:
            csv_file = csv.DictWriter(output_file,
                                      fieldnames=df_list[0].keys(),
                                      )
            csv_file.writeheader()
            csv_file.writerows(df_list)
            print(f'Data succesfully saved to "{save_path}"')

    return df_list

# Testing out with test file of 10 tweets

#keys_to_remove = [
#    'id', 'conversation_id','user_id', 'username', 'name','mentions', 'urls',
#    'photos', 'replies_count', 'retweets_count', 'likes_count',
#    'cashtags', 'link', 'retweet', 'quote_url', 'video', 'thumbnail', 'near',
#    'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date',
#    'translate', 'trans_src', 'trans_dest'
#]

# datacleaning('test.json', keys_to_remove = keys_to_remove)




########### Keys before cleaning ###########
#['id', 'conversation_id', 'created_at', 'date', 'time', 'timezone', 'user_id',
# 'username', 'name', 'place', 'tweet', 'language', 'mentions', 'urls', 'photos',
# 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link',
# 'retweet', 'quote_url', 'video', 'thumbnail', 'near', 'geo', 'source',
# 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date', 'translate',
# 'trans_src', 'trans_dest']

########### Keys after cleaning ###########
#['created_at', 'date', 'time', 'timezone', 'place', 'tweet', 'language', 'hashtags']