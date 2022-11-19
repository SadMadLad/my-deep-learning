import keras
import os

from generator import Generator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

base_dir = os.path.dirname(os.path.abspath(__file__))

def Retrain():
    model = keras.models.load_model(os.path.join(base_dir, "models", "conv_model_retrained.h5"))

    train_gen, validation_gen = Generator()
    batch_size = 32
    epochs = 100

    callback_best_valid = ModelCheckpoint(
        filepath=os.path.join(base_dir, "models", "conv_model_retrained.h5"),
        save_best_only=True,
    )
    callback_early_stopping = EarlyStopping(
        patience=15,
        restore_best_weights=True,
    )

    model.fit(
        train_gen,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_gen,
        shuffle=True,
        callbacks=[callback_best_valid, callback_early_stopping]
    )
    return

Retrain()