import train
import preprocessing


def run():
    preprocessing.config.paths = ['depressive1.json',
                                  'depressive2.json',
                                  'depressive3.json',
                                  'non-depressive1.json',
                                  'non-depressive2.json',
                                  'non-depressive3.json']

    preprocessing.config.labels = ['depressive', 'depressive','depressive',
                                   'not-depressive','not-depressive','not-depressive']

    preprocessing.config.keywords = ['depressed', 'loneliness','hatemyself',
                                     'happiness', 'joy', 'health']

    preprocessing.config.nr_of_tweets = [3000, 3000, 3000, 3000, 3000, 3000]

    history, early_stop_check = train.train_rnn(save_path='./initial_test.pth', collect=False) # Collect=False if already collected data

    #train.show_progress(history=history, save_name='./training_progress.png')
    train.animate_progress(history=history, save_path='./training_animation_progress.gif',
                           early_stop_check=early_stop_check)


run()