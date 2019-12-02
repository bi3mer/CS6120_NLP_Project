from .Utility import log

import pickle
import os

class Model:
    def __init__(self):
        log.info('Instantiating model')
        model_path = os.path.join('models', 'Ridge_Regression_tfidf_sampled.model')
        f = open(model_path, 'rb')
        self.model = pickle.load(f)
        f.close()

        log.info('instantiating data transformer')
        transformer_path = os.path.join('models', 'TF_IDF_Transformer_Sampled')
        f = open(transformer_path, 'rb')
        self.transformer = pickle.load(f)
        f.close()
        
    def score(self, sequences):
        if type(sequences) == str:
            sequences = [sequences]

        log.info('Tokenizing input')
        transforms = self.transformer.transform(sequences)

        log.info('Scoring toxicity of input')
        return self.model.predict(transforms)
