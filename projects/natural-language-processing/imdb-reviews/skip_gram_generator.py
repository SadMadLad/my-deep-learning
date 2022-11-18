from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
tf.device('cpu:0')
import numpy as np

from preprocess import TextsAndLabels

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def GenerateSequences():
    texts, _ = TextsAndLabels()
    texts = np.array(texts)

    print("Number of reviews: ", texts)

    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    return sequences
