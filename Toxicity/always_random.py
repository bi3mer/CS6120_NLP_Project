from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import random
import pickle
import torch
import os

import config

random.seed(0)

path = os.path.join(config.BASE_DIR, 'unsampled_testing.csv')
df = pd.read_csv(path, header=0, usecols=['target'])

y_true = []
y_guess = []

for line in df.iterrows():
    y_true.append(line[1][0])
    y_guess.append(random.random())

print(f'MSE: {mean_squared_error(np.array(y_true), np.array(y_guess))}')