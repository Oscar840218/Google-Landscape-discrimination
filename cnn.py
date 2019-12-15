import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import matplotlib.pyplot as plt


class CNN:
    def __init__(self, epochs, input_shape, classes):
        # initial CNN
        self.EPOCHS = epochs
        self.INPUT_SHAPE = input_shape
        self.CLASSES = classes
        self.chanDim = -1
        self.classifier = Sequential()
        self.detail = None

    def build_model(self):
        # CNN 1
        self.classifier.add(Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            input_shape=self.INPUT_SHAPE,
            activation='relu'
        ))
        self.classifier.add(BatchNormalization(axis=self.chanDim))
        self.classifier.add(MaxPooling2D(pool_size=(3, 3)))
        self.classifier.add(Dropout(0.25))
        # CNN 2
        self.classifier.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ))
        self.classifier.add(BatchNormalization(axis=self.chanDim))
        self.classifier.add(Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ))
        self.classifier.add(BatchNormalization(axis=self.chanDim))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Dropout(0.25))
        # CNN 3
        self.classifier.add(Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        ))
        self.classifier.add(BatchNormalization(axis=self.chanDim))
        self.classifier.add(Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding='same',
            input_shape=self.INPUT_SHAPE,
            activation='relu'
        ))
        self.classifier.add(BatchNormalization(axis=self.chanDim))
        self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
        self.classifier.add(Dropout(0.25))

        # Flatten
        self.classifier.add(Flatten())

        # Full Connection
        self.classifier.add(Dense(1024, activation='relu'))
        self.classifier.add(BatchNormalization(axis=self.chanDim))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(self.CLASSES, activation='softmax'))

        # optimizers
        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adagrad = optimizers.Adagrad()
        #nadam = optimizers.Nadam()

        # Compiling
        self.classifier.compile(optimizer=adagrad, loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.classifier.summary())

    def fit(self, training_data, validation_data):
        print('Start training...')
        self.detail = self.classifier.fit_generator(
            training_data,
            epochs=self.EPOCHS,
            validation_data=validation_data)

    def draw_curves(self):
        if self.detail:
            #
            plt.plot(np.arange(0, self.EPOCHS), self.detail.history["accuracy"], label="training_accuracy")
            #
            plt.plot(np.arange(0, self.EPOCHS), self.detail.history["val_accuracy"], label="testing_accuracy")
            plt.title("Accuracy on Dataset")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower right")
            plt.show()
        else:
            print('Detail data not existed!')
