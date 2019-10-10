from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

import pandas as pd
import random
import pickle
import nltk
import os

import config
import log

training_data_path = os.path.join('.', 'data', 'training_data.pkl')
testing_data_path = os.path.join('.', 'data', 'testing_data.pkl')

split_percentage = 0.8

print('------------------------------------------------------------------')
log.info('Downloading or updating Punkt for word tokenization')
nltk.download('punkt')
print('------------------------------------------------------------------')

log.info('Loading Training Data')
# df = pd.read_csv(config.TRAIN_PATH, header=0, nrows=10000, usecols=['target', 'comment_text'])
df = pd.read_csv(config.TRAIN_PATH, header=0, usecols=['target', 'comment_text'])

log.info('Converting text to lower case')
targets = df['comment_text'].apply(lambda comment: comment.lower())
df.update(targets)

log.info('Tokenizing text and formatting training data')
tweet_tokenizer = nltk.tokenize.TweetTokenizer()
data = df.to_dict()
data_set = []

for key in tqdm(data['target'], ascii=True):
    data_set.append((tweet_tokenizer.tokenize(data['comment_text'][key]), data['target'][key]))

log.info(f'Splitting data with {split_percentage * 100}% as part of the training data')
training_set, testing_set = train_test_split(data_set, test_size = 1.0 - split_percentage)

log.info(f'Train Data Size: {len(training_set)}')
log.info(f'Test Data Size: {len(testing_set)}')

log.info(f'Saved training data to {training_data_path}')
log.info(f'Saved testing data to {testing_data_path}')