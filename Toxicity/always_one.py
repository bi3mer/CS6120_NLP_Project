from sklearn.metrics import mean_squared_error

import numpy as np
import pickle
import torch
import os

import pandas as pd
import config

path = os.path.join(config.BASE_DIR, 'unsampled_testing.csv')
df = pd.read_csv(path, header=0, usecols=['target'])


y_true = []


for line in df.iterrows():
    y_true.append(line[1][0])

y_guess = np.full(len(y_true), 1)
print(f'MSE: {mean_squared_error(np.array(y_true), y_guess)}')