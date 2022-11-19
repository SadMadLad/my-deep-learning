from keras import layers
from keras.models import Model


def ConvModel(dim_1, dim_2):
    activation = "selu"
    inputs = layers.Input(shape=(dim_1, dim_2, 1))

    conv = layers.Conv2D(8, (3, 3), strides=(
        1, 1), activation=activation)(inputs)
    conv = layers.MaxPooling2D()(conv)
    conv = layers.BatchNormalization()(conv)

    conv = layers.Conv2D(16, (3, 3), strides=(
        1, 1), activation=activation)(conv)
    conv = layers.MaxPooling2D()(conv)
    conv = layers.BatchNormalization()(conv)

    conv = layers.Flatten()(conv)
    conv = layers.Dense(16, activation=activation)(conv)
    outputs = layers.Dense(10, activation="softmax")(conv)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def CompiledConvModel(dim_1, dim_2, optimizer, loss, metrics):
    model = ConvModel(dim_1, dim_2)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model
