import tensorflow as tf
import random
import nltk

from .Utility import log

class Model:
    def __init__(self):
        log.info('Instantiating model')

    def score(self, sequence):
        log.info('Tokenizing and scoring sequence')

        return random.random()