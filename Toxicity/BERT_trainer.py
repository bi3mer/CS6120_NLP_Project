#!/usr/bin/env python
# coding: utf-8

# ## Configuration

# #### Model



# model_type = 'bert-base-cased'
model_type = 'bert-base-uncased'
# model_type = 'bert-large-cased'
# model_type = 'bert-large-uncased'

dataset_size = None # set to None for full dataset
min_length = 140


# #### Learning Parameters



epochs = 10
learning_rate = 2e-5
warmup = 0.05
batch_size = 32
accumulation_steps=2
seed = 0


# ## Variables to Not Change



max_sentence_length = 512
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']




if dataset_size == None:
    output_model_file = f'{model_type}.bin'
else:
    output_model_file = f'{dataset_size}_{model_type}.bin'


# ## Check Configuration

# This is kept pretty naive. Mainly want to make sure that a model isn't overwritten.



import os

if not os.path.isdir('model'):
    os.mkdir('model')

model_output_path = os.path.join('model', output_model_file)
assert os.path.exists(model_output_path) == False


# ## Getting Data for BERT



from torch.utils.data import TensorDataset

import numpy as np
import pickle
import torch




data_path = f'{model_type}_{dataset_size}'




f = open(os.path.join('data', f'{data_path}_training_data.pkl'), 'rb')
x, y = pickle.load(f)
f.close()




y = torch.tensor([torch.tensor(_y, dtype=torch.float) for _y in y])




new_x = []
for row in x:
    while len(row) < min_length:
        row.append(0)
        
    new_x.append(x)




dataset = TensorDataset(torch.tensor(x, dtype=torch.long), y)


# ## Loading Bert



from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np




torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)




tokenizer = BertTokenizer.from_pretrained(model_type)


# ## Load Pre-Trained BERT Model



from pytorch_pretrained_bert import BertForSequenceClassification,BertAdam


print('loading model')
model = BertForSequenceClassification.from_pretrained(model_type,cache_dir=None,num_labels=1)

# ## Fine-Tune BERT

from torch.nn import functional as F
from tqdm import tqdm, trange




train_optimization_steps = int(epochs*len(dataset)/batch_size/accumulation_steps)




param_optimizer = list(model.named_parameters())

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]




optimizer = BertAdam(
    optimizer_grouped_parameters,
    lr=learning_rate,
    warmup=warmup,
    t_total=train_optimization_steps)



criterion = torch.nn.MSELoss()  
model = model.train()


for _ in trange(epochs, desc='epoch'):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer.zero_grad()

    for step, (x, y) in tqdm(enumerate(train_loader), desc='batch'):
        predictions = model(x)
        
        loss = criterion(predictions, y)
        
        loss.backward()
        optimizer.step()        
        optimizer.zero_grad()

torch.save(model.state_dict(), model_output_path)