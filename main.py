import train
import preprocessing


def run():
    '''
    Training function to run the training process after specifying parameters
    '''

    preprocessing.config.paths = ['./training_data/depressive1.json',
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

    preprocessing.config.save_path = './training_data/all_training_data.csv'


    preprocessing.config.labels = ['depressive', 'depressive', 'depressive', 'depressive', 'depressive', 'depressive',
                                   'not-depressive', 'not-depressive', 'not-depressive', 'not-depressive',
                                   'not-depressive', 'not-depressive']

    preprocessing.config.keywords = ['depressed', 'lonely', 'sad', 'depression', 'tired', 'anxious',
                                     'happy', 'joy', 'thankful', 'health', 'hopeful', 'glad']

    preprocessing.config.nr_of_tweets = [1000, 1000, 1000, 1000, 1000, 1000,
                                         1000, 1000, 1000, 1000, 1000, 1000]

    history, early_stop_check = train.train_rnn(save_path='./weights/lstm_model_2.pth', collect=True) # Collect=False if already collected data

    train.show_progress(history=history, save_name='./plots/training_progress.png')

    train.animate_progress(history=history, save_path='./plots/training_animation_progress_REAL.gif',
                           early_stop_check=early_stop_check)

run()