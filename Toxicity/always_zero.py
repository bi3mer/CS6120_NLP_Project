from sklearn.metrics import mean_squared_error

import numpy as np
import pickle
import torch
import os

f = open(os.path.join('data', 'bert-base-cased_testing_data.pkl'), 'rb')
test = pickle.load(f)
f.close()

y_true = []

for line in test:
    y_true.append(line[2])

y_guess = np.zeros(len(y_true))
print(f'MSE: {mean_squared_error(np.array(y_true), y_guess)}')