from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
tf.device('cpu:0')
import numpy as np

from preprocess import TextsAndLabels

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def TokenizerInfo(tokenizer, texts):
    vocab_words = len(tokenizer.word_index.items())+1

    print("Total Vocabulary Words: ", vocab_words)

    print("Top Words: ")
    print("\t", dict(list(tokenizer.word_index.items())[:10]))
    print("Bottom Words: ")
    print("\t", dict(list(tokenizer.word_index.items())[-10:]))

    print("\nOriginal Text: ", texts[0][:100])
    print(f"Sequence IDs: ", tokenizer.texts_to_sequences(
        [texts[0][:100]])[0])

    return vocab_words


def SkipGramInfo(tokenizer, texts_sequences, inputs, labels):
    sample_word_ids = texts_sequences[0][:5]
    sample_phrase = ' '.join([tokenizer.index_word[wid]
                             for wid in sample_word_ids])

    print("\nSample Phrase: ", sample_phrase)
    print("Sample Word IDs: ", sample_word_ids)

    print("\nSample Skip Grams: ")
    for inp, lbl in zip(inputs, labels):
        print(
            f"\tInput: {inp}, Words: {[tokenizer.index_word[wi] for wi in inp]}, Label: {lbl}")

    return


def Tokenize():
    texts, ratings = TextsAndLabels()
    texts = np.array(texts)
    print(len(texts))
    tokenizer = Tokenizer(lower=True, split=" ")

    tokenizer.fit_on_texts(texts)
    vocab_words = TokenizerInfo(tokenizer, texts)

    texts_sequences = tokenizer.texts_to_sequences(texts)

    sample_word_ids = texts_sequences[0][:5]
    inputs, labels = skipgrams(
        sequence=sample_word_ids,
        vocabulary_size=vocab_words,
        window_size=1,
        negative_samples=0,
        shuffle=False,
    )
    SkipGramInfo(tokenizer, texts_sequences, inputs, labels)
    inputs, labels = np.array(inputs), np.array(labels)

    # The negative sample generated by this skipgrams can lead to poor performance
    # Now, we will generate our own negative samples

    print(inputs[:1, 1:])

    negative_sampling_candidates, true_expected_count, sampled_expected_count = \
        tf.random.log_uniform_candidate_sampler(
            true_classes=inputs[:1, 1:],
            num_true=1,
            num_sampled=10,
            unique=True,
            range_max=vocab_words,
            name="negative_sampling"
        )

    return

# Tokenize()
