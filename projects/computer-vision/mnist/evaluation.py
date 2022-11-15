import os
import keras
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
# Custom Modules
from preprocess import Preprocess

base_dir = os.path.dirname(os.path.abspath(__file__))

dense_model = os.path.join(base_dir, "models", "dense_model.h5")
conv_model = os.path.join(base_dir, "models", "conv_model.h5")

dense_model = keras.models.load_model(dense_model)
conv_model = keras.models.load_model(conv_model)


def EvaluationPipe():
    preprocess = Preprocess()

    dataset_path = os.path.join(base_dir, "dataset", "fashion-mnist_test.csv")
    df = pd.read_csv(dataset_path)

    images, labels = preprocess.DatasetToNumpy(df), preprocess.ReturnLabels(df)
    labels = labels.to_numpy()

    images = images / 255
    shape = images.shape
    images_2d = images.reshape(shape[0], 28, 28, 1)

    print("Images Flattened: ", images.shape)
    print("Images 2D: ", images_2d.shape)
    print("Labels: ", labels.shape, "\n")

    return images, images_2d, labels


images_flat, images_2d, labels = EvaluationPipe()


def EvaluateModel(model, model_name, inputs, labels):
    predictions = model.predict(inputs)
    predictions = np.argmax(predictions, 1)

    print(model_name)
    print("Accuracy of Model: ", accuracy_score(predictions, labels), "\n")


EvaluateModel(dense_model, "Dense Model", images_flat, labels)
EvaluateModel(conv_model, "Conv Model", images_2d, labels)
