from pytorch_pretrained_bert import BertTokenizer
from collections import Counter
from tqdm import tqdm

import pandas as pd
import random
import pickle
import os

from Utility import log
import config

def tokenize_df(df, tokenizer):
    data = df.to_dict()
    x = []
    y = []

    for key in tqdm(data['target'], ascii=True):
        # max input is 512 for BERT network and we have to one at the start and one at the end
        tokenized_text = tokenizer.tokenize(f'[CLS] {data["comment_text"][key]}')[:139]
        tokenized_text.append('[SEP]')
        tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

        x.append(tokenized_ids)
        y.append(data['target'][key])

    return [x, y]

def process_data(model_type, size=None):
    training_data_path = os.path.join('.', 'data', f'{model_type}_{size}_training_data.pkl')
    testing_data_path = os.path.join('.', 'data', f'{model_type}_{size}_testing_data.pkl')

    log.info('Loading Training Data')
    path = os.path.join(config.BASE_DIR, 'sampled_training.csv')
    train_df = pd.read_csv(path, header=0, nrows=size, usecols=['target', 'comment_text'])

    log.info('loading testing data')
    path = os.path.join(config.BASE_DIR, 'unsampled_testing.csv')
    test_df = pd.read_csv(path, header=0, nrows=size, usecols=['target', 'comment_text'])

    if 'uncased' in model_type:
        log.info('Converting training text to lower case')
        targets = train_df['comment_text'].apply(lambda comment: comment.lower())
        train_df.update(targets)

        log.info('Converting testing text to lower case')
        targets = test_df['comment_text'].apply(lambda comment: comment.lower())
        test_df.update(targets)

    log.info('loading tokenizer')
    tokenizer = BertTokenizer.from_pretrained(model_type)

    log.info('tokenizing training data')
    training_set = tokenize_df(train_df, tokenizer)

    log.info('tokenizing testing data')
    testing_set = tokenize_df(test_df, tokenizer)

    log.info(f'Train Data Size: {len(train_df)}')
    log.info(f'Test Data Size: {len(test_df)}')

    pickle.dump(training_set, open(training_data_path, 'wb'))
    log.info(f'Saved training data to {training_data_path}')

    pickle.dump(testing_set, open(testing_data_path, 'wb'))
    log.info(f'Saved testing data to {testing_data_path}')

if __name__ == '__main__':
    size = 1000

    process_data('bert-base-cased', size=size)
    process_data('bert-base-uncased', size=size)
    # process_data('bert-large-cased', size=size)
    # process_data('bert-large-uncased', size=size)
