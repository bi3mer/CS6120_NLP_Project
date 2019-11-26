from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

import pandas as pd
import numpy as np
import random
import math
import os

from Utility import log
import config

train_test_split_percentage = 0.8
bins = 50
seed = 0

training_data_path = os.path.join(config.BASE_DIR, 'sampled_training.csv')
testing_data_path = os.path.join(config.BASE_DIR, 'sampled_testing.csv')

log.info('Loading Training Data')
df = pd.read_csv(config.TRAIN_PATH, header=0, usecols=['target', 'comment_text'])

log.info(f'Splitting data with {train_test_split_percentage * 100}% as part of the training data')
training_set, testing_set = train_test_split(df, test_size = 1.0 - train_test_split_percentage, random_state=0)

log.info(f'Train Data Size: {len(training_set)}')
log.info(f'Test Data Size: {len(testing_set)}')

f = open(testing_data_path, 'w')
f.write(df.to_csv(index=False))
f.close()
log.info(f'Saved testing to {testing_data_path}')

log.info('Downsampling the training data. This can take about a minute or three.')
# First we collect all of the data into bins.
x = {}
y = {}

for val in df.iterrows():
    target = val[1][0]
    bin_index = math.floor(target / (1/bins))
    
    if bin_index not in x:
        x[bin_index] = []
        y[bin_index] = []
        
    x[bin_index].append(val[1][1])
    y[bin_index].append(target)

# Now we find the bin with the second most occurrences and the assocaited count.
counter = Counter()

for key in y:
    dictionary = {}
    dictionary[key] = len(y[key])
    counter.update(Counter(dictionary))

most_common = counter.most_common(2)
second_most_common_count = most_common[1][1]

# Now we will randomly sample the most occurring column to reduce it to be no more common 
# than the second most occurring.
most_common_key = most_common[0][0]
most_common_count = most_common[0][1]

random.seed(seed)
indexes = [i for i in range(most_common_count)]
indexes = random.sample(indexes, second_most_common_count)

new_x = []
new_y = []

for key in y:
    if key == most_common_key:
        for index in indexes:
            new_x.append(x[key][index])
            new_y.append(y[key][index])
    else:
        new_x.extend(x[key])
        new_y.extend(y[key])

# Now we can create the dataframe and save the results of our hardwork undersampling.
data = {'target': new_y, 'comment_text': new_x}
df = pd.DataFrame(data)

f = open(training_data_path, 'w')
f.write(df.to_csv(index=False))
f.close()

log.info(f'Saved testing data to {training_data_path}')