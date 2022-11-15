import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

# Custom Modules
from image_manual import ImageHelper
from dense_model import CompiledModel
from conv_model import CompiledConvModel
from preprocess import Preprocess

# Loading Data
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

base_dir = os.path.dirname(os.path.abspath(__file__))


def FlattenDataPipe():
    preprocess = Preprocess()

    dataset_path = os.path.join(base_dir, "dataset", "fashion-mnist_train.csv")

    df = pd.read_csv(dataset_path)

    df, labels = preprocess.DatasetToNumpy(df), preprocess.ReturnLabels(df)
    labels = preprocess.GetLabelDummies(labels)

    df = df/255

    X_train, X_test, Y_train, Y_test = train_test_split(
        df, labels, test_size=0.25, stratify=labels)

    print("\nAfter Preprocessing: \n")
    print("X Train: ", X_train.shape)
    print("X Valid: ", X_test.shape)

    print("Y Train: ", Y_train.shape)
    print("Y Valid: ", Y_test.shape, "\n")

    return X_train, X_test, Y_train, Y_test


def TwoDimensionalDataPipe():
    preprocess = Preprocess()

    dataset_path = os.path.join(base_dir, "dataset", "fashion-mnist_train.csv")

    df = pd.read_csv(dataset_path)

    df, labels = preprocess.DatasetToNumpy(df), preprocess.ReturnLabels(df)
    labels = preprocess.GetLabelDummies(labels)

    df = df/255

    # Reshaping the dataframe
    shape = df.shape
    df = df.reshape(shape[0], 28, 28, 1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        df, labels, test_size=0.25, stratify=labels)

    print("\nAfter Preprocessing: \n")
    print("X Train: ", X_train.shape)
    print("X Valid: ", X_test.shape)

    print("Y Train: ", Y_train.shape)
    print("Y Valid: ", Y_test.shape, "\n")

    return X_train, X_test, Y_train, Y_test

# X_train, X_valid, Y_train, Y_valid = FlattenDataPipe()
X_train, X_valid, Y_train, Y_valid = TwoDimensionalDataPipe()
dense = CompiledModel(28, 28, 'nadam', 'categorical_crossentropy', ['accuracy'])
conv = CompiledConvModel(28, 28, 'nadam', 'categorical_crossentropy', ['accuracy'])


def TrainModel(model, model_name):
    batch_size = 32
    epochs = 50
    validation_set = (X_valid, Y_valid)

    callback_best_valid = ModelCheckpoint(
        filepath=os.path.join(base_dir, "models", model_name),
        save_best_only=True,
    )
    callback_early_stopping = EarlyStopping(
        patience=8,
        restore_best_weights=True,
    )

    model.fit(
        x=X_train,
        y=Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_set,
        shuffle=True,
        callbacks=[callback_best_valid, callback_early_stopping]
    )

TrainModel(conv, "conv_model.h5")
