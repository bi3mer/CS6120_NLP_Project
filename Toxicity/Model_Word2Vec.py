import pickle
import sklearn

from .Utility import log

class Model_Word2Vec:
    def __init__(self):
        log.info('Instantiating word2vec model')
        path_to_model = './models/Ridge_Regression_Word_2_Vec.model'
        with open(path_to_model, 'rb') as file:
            self.model = pickle.load(file)
        model_type = 'word2vec'
        log.info('load word2vec here')
        # self.tokenizer =

    def score(self, sequence):
        log.info('Tokenizing input')
        # tokenized_input = self.tokenizer.tokenize(f'[CLS] {sequence}')[:139]
        # tokenized_input.append('[SEP]')
        #
        # tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_input)
        #
        # while len(tokenized_ids) < 140:
        #     tokenized_ids.append(0)
        #
        # log.info('Scoring toxicity of input')
        # with torch.no_grad():
        #     prediction = self.model.forward(torch.tensor([tokenized_ids], dtype=torch.long))
        #     return prediction[0]

        # log.error("There was an error calculating the toxicity with the model.")
        return 1.5
