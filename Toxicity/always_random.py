from sklearn.metrics import mean_squared_error
<<<<<<< HEAD

import pandas as pd
import numpy as np
import random
=======
from random import random

import numpy as np
>>>>>>> ae37d140d75d4815e58a5214c8406e2d26b38ff3
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