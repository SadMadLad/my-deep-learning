from skip_gram_generator import Generator

from keras.models import Model
from keras import layers


def EmbeddingModel(vocabulary_size, embedding_size):
    target = layers.Input(shape=(), name="target")
    context = layers.Input(shape=(), name="context")

    target = layers.Embedding(
        input_dim = vocabulary_size,
        output_dim = embedding_size,
        name="target_embedding"
    )(target)
    context = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_size,
        name="context_embedding"
    )(context)

    