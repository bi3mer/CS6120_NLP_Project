from pytorch_pretrained_bert import BertForSequenceClassification
from pytorch_pretrained_bert import BertTokenizer

import torch
import os

from .Utility import log

class Model:
    def __init__(self):
        log.info('Instantiating model')
        model_type = 'bert-base-cased'
        self.model = BertForSequenceClassification.from_pretrained(model_type, cache_dir=None,num_labels=1)

        log.info('Loading model to CPU')
        device = torch.device('cpu')
        path = os.path.join('Toxicity', 'model', f'None_{model_type}.bin')
        state_dict = torch.load(path, map_location=device)
        self.model.load_state_dict(state_dict)
        
        log.info('Instantiating tokenizer')
        self.tokenizer = BertTokenizer.from_pretrained(model_type)

    def score(self, sequence):
        log.info('Tokenizing input')
        tokenized_input = self.tokenizer.tokenize(f'[CLS] {sequence}')[:139]
        tokenized_input.append('[SEP]')

        tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_input)

        while len(tokenized_ids) < 140:
            tokenized_ids.append(0)

        log.info('Scoring toxicity of input')
        with torch.no_grad():
            prediction = self.model.forward(torch.tensor([tokenized_ids], dtype=torch.long))
            return prediction[0]

        log.error("There was an error calculating the toxicity with the model.")
        return 0

    def score_multiple(self, sequences):
        log.info('Tokenizing input')
        tokenized_ids = []
        for sequence in sequences:
            tokenized_input = self.tokenizer.tokenize(f'[CLS] {sequence}')[:139]
            tokenized_input.append('[SEP]')

            ids = self.tokenizer.convert_tokens_to_ids(tokenized_input)

            while len(ids) < 140:
                ids.append(0)
            tokenized_ids.append(ids)

        log.info('scoring toxicity')
        with torch.no_grad():
            return self.model.forward(torch.tensor(tokenized_ids, dtype=torch.long))

        log.error('Encountered error calculating toxicity with the model')
        return [0 for _ in range(len(sequences))]