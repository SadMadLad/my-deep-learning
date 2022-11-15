import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

# Custom Modules
from dense_model import CompiledModel
from preprocess import Preprocess

# Loading Data
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

base_dir = os.path.dirname(os.path.abspath(__file__))


def DataPipe():
    preprocess = Preprocess()

    dataset_path = os.path.join(base_dir, "dataset", "fashion-mnist_train.csv")
    test_path = os.path.join(base_dir, "dataset", "fashion-mnist_test.csv")

    df = pd.read_csv(dataset_path)
    df_test = pd.read_csv(test_path)

    df, labels = preprocess.DatasetToNumpy(df), preprocess.ReturnLabels(df)
    labels = preprocess.GetLabelDummies(labels)
    df_test, test_labels = preprocess.DatasetToNumpy(
        df_test), preprocess.ReturnLabels(df_test)
    test_labels = preprocess.GetLabelDummies(test_labels)

    X_train, X_test, Y_train, Y_test = train_test_split(
        df, labels, test_size=0.25, stratify=labels)

    print("\nAfter Preprocessing: \n")
    print("X Train: ", X_train.shape)
    print("X Valid: ", X_test.shape)
    print("X Test: ", df_test.shape, "\n")

    print("Y Train: ", Y_train.shape)
    print("Y Valid: ", Y_test.shape)
    print("Y Test: ", test_labels.shape, "\n")

    return X_train, X_test, df_test, Y_train, Y_test, test_labels


X_train, X_valid, X_test, Y_train, Y_valid, Y_test = DataPipe()


def TrainDenseModel():
    model = CompiledModel(
        28, 28, 'nadam', 'categorical_crossentropy', ['accuracy'])

    batch_size = 32
    epochs = 50
    validation_set = (X_valid, Y_valid)

    callback_best_valid = ModelCheckpoint(
        filepath=os.path.join(base_dir, "models", "dense_model.h5"),
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


TrainDenseModel()
