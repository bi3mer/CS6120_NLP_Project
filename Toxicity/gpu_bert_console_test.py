from pytorch_pretrained_bert import BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

import pickle
import torch
import os

def test_model(model, test, batch_size):
    with torch.no_grad():
        loader = DataLoader(test, batch_size=batch_size)
        mse = torch.nn.MSELoss()
        mses = []
        
        for _, (x, y) in enumerate(loader):
            predictions = model.forward(x)
            mses.append(mse(predictions, y))
            print(f'batch MSE: {mses[-1]}')
            
        print(f'MSE: {sum(mses) / float(len(mses))}')

def spread_data(dataset):
    x = []
    y = []
    split = []
    
    for val in dataset:
        x.append(val[0])
        split.append(val[1])
        y.append(val[2])
        
    y = torch.tensor(y, dtype=torch.float)
    split = torch.tensor(split)
    
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float), torch.tensor(split)

# configuration variables
model_size = 1000 # set to None for full dataset
data_set_size = None
batch_size = 32
model_type = 'bert-base-cased'

# variables based on configuration
if model_size == None:
    model_file_name = f'cased_base.bin'
else:
    model_file_name = f'{model_size}_cased_base.bin'

if data_set_size == None:
    test_pickle_file_name = f'{model_type}_testing_data.pkl'
else:
    test_pickle_file_name = f'{model_type}_{data_set_size}_testing_data.pkl'

# load BERT
model = BertForSequenceClassification.from_pretrained(model_type,cache_dir=None,num_labels=1)
model.load_state_dict(torch.load(os.path.join('model', model_file_name)))

# load in test data set
f = open(os.path.join('data', test_pickle_file_name), 'rb')
test = pickle.load(f)
f.close()

test_x, test_y, split_indexes_test = spread_data(test)
test = TensorDataset(test_x, test_y)

# run test
test_model(model, test, batch_size)