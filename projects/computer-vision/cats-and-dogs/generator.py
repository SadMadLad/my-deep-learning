import os
from keras.preprocessing.image import ImageDataGenerator

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_dir, "dataset")

print(dataset_path)


def Generator():
    augmentor = ImageDataGenerator(
        horizontal_flip=True,
        rescale=1.0/255,
        validation_split=0.2,
        rotation_range=45,
    )
    train_generator = augmentor.flow_from_directory(
        dataset_path,
        target_size=(256, 256),
        color_mode='grayscale',
        classes=['dogs', 'cats'],
        batch_size=32,
        shuffle=True,
        class_mode='binary',
        subset='training',
    )
    validation_generator = augmentor.flow_from_directory(
        dataset_path,
        target_size=(256, 256),
        color_mode='grayscale',
        classes=['dogs', 'cats'],
        batch_size=32,
        shuffle=True,
        class_mode='binary',
        subset='validation',
    )
    return train_generator, validation_generator
