from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer
from collections import Counter
from tqdm import tqdm

import pandas as pd
import random
import pickle
import os

from Utility import log
import config

def process_data(model_type, size=None):
    if size == None:
        training_data_path = os.path.join('.', 'data', f'{model_type}_training_data.pkl')
        testing_data_path = os.path.join('.', 'data', f'{model_type}_testing_data.pkl')
    else:
        training_data_path = os.path.join('.', 'data', f'{model_type}_{size}_training_data.pkl')
        testing_data_path = os.path.join('.', 'data', f'{model_type}_{size}_testing_data.pkl')

    split_percentage = 0.8

    log.info('Loading Training Data')
    if size == None:
        df = pd.read_csv(config.TRAIN_PATH, header=0, usecols=['target', 'comment_text'])
    else:
        df = pd.read_csv(config.TRAIN_PATH, header=0, nrows=size, usecols=['target', 'comment_text'])

    if 'uncased' in model_type:
        log.info('Converting text to lower case')
        targets = df['comment_text'].apply(lambda comment: comment.lower())
        df.update(targets)

    log.info('Tokenizing text and formatting training data')
    tokenizer = BertTokenizer.from_pretrained(model_type)
    data = df.to_dict()
    data_set = []

    for key in tqdm(data['target'], ascii=True):
        tokenized_text = tokenizer.tokenize(data['comment_text'][key])[:512]
        tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        
        mid_point = len(tokenized_ids) / 2
        segment_ids = [0 if i < mid_point else 1 for i in range(len(tokenized_ids))]

        data_set.append((tokenized_ids, segment_ids, data['target'][key]))

    log.info(f'Splitting data with {split_percentage * 100}% as part of the training data')
    training_set, testing_set = train_test_split(data_set, test_size = 1.0 - split_percentage)

    log.info(f'Train Data Size: {len(training_set)}')
    log.info(f'Test Data Size: {len(testing_set)}')

    pickle.dump(training_set, open(training_data_path, 'wb'))
    log.info(f'Saved training data to {training_data_path}')

    pickle.dump(testing_set, open(testing_data_path, 'wb'))
    log.info(f'Saved testing data to {testing_data_path}')

if __name__ == '__main__':
    size = 1000

    process_data('bert-base-cased', size=size)
    process_data('bert-base-uncased', size=size)
    process_data('bert-large-cased', size=size)
    process_data('bert-large-uncased', size=size)