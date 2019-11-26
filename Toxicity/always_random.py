from sklearn.metrics import mean_squared_error
from random import random

import numpy as np
import pickle
import torch
import os

f = open(os.path.join('data', 'bert-base-cased_testing_data.pkl'), 'rb')
test = pickle.load(f)
f.close()

y_true = []
y_guess = []

for line in test:
    y_true.append(line[2])
    y_guess.append(random())

print(f'MSE: {mean_squared_error(np.array(y_true), np.array(y_guess))}')