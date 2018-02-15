import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import pandas as pd


def build_cnn_fn():
    # Initialize classifier object
    classifier = Sequential()

    # Add first convolutional layer
    classifier.add(Convolution2D(input_shape=(64, 64, 3),
                                 filters=32,
                                 kernel_size=[3, 3],
                                 strides=2,
                                 activation='relu'))

    # Add first max pooling layer
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Add second convolutional layer
    classifier.add(Convolution2D(filters=32,
                                 kernel_size=[3, 3],
                                 strides=2,
                                 activation='relu'))

    # Add second max pooling layer
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten feature map
    classifier.add(Flatten())

    # Add first fully connected layer
    classifier.add(Dense(units=128,
                         activation='relu'))

    # Add final fully connected layer
    classifier.add(Dense(units=1,
                         activation='sigmoid'))

    # Compile classifier
    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    # Return classifier
    return classifier


def augment_image(training_dir, testing_dir):
    train_datagen = ImageDataGenerator(
            rescale=1./255,  # Rescale to 0-1
            shear_range=0.2,  # Apply random transvections
            zoom_range=0.2,  # Apply random zooms
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_set = train_datagen.flow_from_directory(
            training_dir,
            target_size=(64, 64),  # New image sizes
            batch_size=32,
            class_mode='binary')

    test_set = test_datagen.flow_from_directory(
            testing_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

    return train_set, test_set


def fit_model(model_object, train_set, test_set):
    model_object.fit_generator(
        train_set,
        steps_per_epoch=8000,  # Number of images in training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)  # Number of images in test set


if __name__ == '__main__':
    model = build_cnn_fn()
    train_data, test_data = augment_image(
        training_dir='Datasets/training_set',
        testing_dir='Datasets/test_set')

    fit_model(model, train_data, test_data)
