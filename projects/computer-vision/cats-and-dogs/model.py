from keras import layers
from keras.models import Model

def ConvModel(dim_1, dim_2):
    activation = 'swish'

    inputs = layers.Input(shape=(dim_1, dim_2, 1))

    conv = layers.Conv2D(4, kernel_size=(3, 3), strides=(1, 1), activation=activation)(inputs)
    conv = layers.MaxPooling2D()(conv)
    conv = layers.BatchNormalization()(conv)

    conv = layers.Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation=activation)(conv)
    conv = layers.MaxPooling2D()(conv)
    conv = layers.BatchNormalization()(conv)

    conv = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation=activation)(conv)
    conv = layers.MaxPooling2D()(conv)
    conv = layers.BatchNormalization()(conv)

    conv = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation=activation)(conv)
    conv = layers.MaxPooling2D()(conv)
    conv = layers.BatchNormalization()(conv)

    conv = layers.Flatten()(conv)
    conv = layers.Dense(256, activation=activation)(conv)
    conv = layers.BatchNormalization()(conv)
    outputs = layers.Dense(1, activation='sigmoid')(conv)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def CompiledModel(dim_1, dim_2, optimizer, loss, metrics):
    model = ConvModel(dim_1, dim_2)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

# CompiledModel(256, 256, 'nadam', 'binary_crossentropy', ['accuracy']).summary()