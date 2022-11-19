from keras import layers
from keras.models import Model


def DenseModel(dim_1, dim_2):
    activation = "selu"

    inputs = layers.Input(shape=(dim_1 * dim_2, 1))
    flatten = layers.Flatten()(inputs)

    dense = layers.Dense(56, activation=activation)(flatten)
    dense = layers.BatchNormalization()(dense)

    dense = layers.Dense(14, activation=activation)(dense)
    outputs = layers.Dense(10, activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def CompiledModel(dim_1, dim_2, optimizer, loss, metrics):
    model = DenseModel(dim_1, dim_2)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics
    )
    return model
