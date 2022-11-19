from preprocess import TextsAndLabels

import os
import numpy as np
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.sequence import make_sampling_table
from keras.preprocessing.text import Tokenizer
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.device('cpu:0')

"""Skipgram: Predict context words from target word."""


def GenerateSequences():
    texts, _ = TextsAndLabels()
    texts = np.array(texts)

    print("Number of reviews: ", len(texts))

    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    vocabulary_size = len(tokenizer.word_index.items())+1
    print("Vocabulary Size: ", vocabulary_size)

    return sequences, tokenizer, vocabulary_size


def SkipgramGenerator(sequences, vocabulary_size, seed=1):
    """
    Note: Array can be a tensor or a numpy array.

    Shape Variables:
    M: The number of vocabulary words for the specific sequence.
    N: Number of negative samples
    k: Any index value ranged in (0, 63147).
    VAR: Variable

    Variables           Type                Shape               Description

    vocabulary_size     (int)               63174               Total Number of Words to learn
    sequences           (List[List[int]])   (45781, M)          List of embeddings for each review
    rand_sequence_ids   (Array[int])        (45781, )           Shuffled array of indexes for each review
    sampling_table      (Array[int])        (63147, )           A word rank-based probabilistic sampling table

    si                  (int)               k                   An index of rand_sequence_ids.
    positive_skip_grams (List[List[int]])   (VAR, 2)            Pair of word indexes that occur in each other's context
    target_word         (int)               k                   Central Word Embedding
    context_word        (int)               k                   Surrounding words of the target_word

    context_class       (Array[Array[int]]) (1, 1)              To pass through for negative sampling
    context             (Array[int])        (N+1, )             One surrounding plus N negative samples 
    label               (Array[int])        (N+1, )             One + N-zeroes to label the surrounding context
    target              (List[int])         (N, )               N-times the embedding index of target_word.
    contexts
    labels
    targets

    """
    # Arguments
    window_size = 1
    negative_samples = 4
    batch_size = 1000

    rand_sequence_ids = np.arange(len(sequences))
    np.random.shuffle(rand_sequence_ids)

    sampling_table = make_sampling_table(
        size=vocabulary_size, sampling_factor=1e-05)

    for si in rand_sequence_ids:
        # print("Sequence Shape: ", len(sequences[si]))
        positive_skip_grams, _ = skipgrams(
            sequence=sequences[si],
            vocabulary_size=vocabulary_size,
            shuffle=False,
            window_size=window_size,
            negative_samples=0,
            sampling_table=sampling_table,
        )
        # print("Positive Skip Gram Length: ", len(positive_skip_grams))
        targets, contexts, labels = [], [], []
        for target_word, context_word in positive_skip_grams:
            # print("target_word: ", target_word)
            # print("context_word: ", context_word)

            context_class = tf.expand_dims(tf.constant(
                [context_word], dtype="int64"), axis=1)

            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=negative_samples,
                unique=True,
                range_max=vocabulary_size,
                seed=seed,
                name="negative_sampler"
            )

            context = tf.concat([tf.constant(
                [context_word], dtype="int64"), negative_sampling_candidates], axis=0)
            label = tf.constant([1] + [0]*negative_samples, dtype="int64")
            target = [target_word]*(negative_samples+1)

            targets.extend(target)
            contexts.append(context)
            labels.append(label)

        try:
            contexts, targets, labels = np.concatenate(contexts), \
                np.array(targets), np.concatenate(labels)
        except:
            continue

        assert contexts.shape[0] == targets.shape[0]
        assert contexts.shape[0] == labels.shape[0]

        np.random.seed(seed)
        np.random.shuffle(contexts)
        np.random.seed(seed)
        np.random.shuffle(targets)
        np.random.seed(seed)
        np.random.shuffle(labels)

        for eg_id_start in range(0, contexts.shape[0], batch_size):
            print("In batch")
            yield (
                targets[eg_id_start: min(
                    eg_id_start+batch_size, targets.shape[0])],
                contexts[eg_id_start: min(
                    eg_id_start+batch_size, contexts.shape[0])]
            ), labels[eg_id_start: min(eg_id_start+batch_size, labels.shape[0])]

def Generator():
    sequences, tokenizer, vocabulary_size = GenerateSequences()
    skipgram_generator = SkipgramGenerator(sequences, vocabulary_size)
    return vocabulary_size, tokenizer, skipgram_generator