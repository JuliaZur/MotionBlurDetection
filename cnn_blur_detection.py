from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from data_generator import get_regions_with_labels
from data_generator import BlurRegionsDataGenerator
from tensorflow import keras
from data_preparation import BLUR_COLS, BLUR_ROWS, get_img_regions
import numpy as np


class MotionBlurDetectionCNN:

    def __init__(self, input_shape=(30, 30, 3), classes_number=2, epochs=10, learning_rate=0.01, batch_size=64,
                 test_size=0.2):
        self.input_shape = input_shape
        self.classes_number = classes_number
        self.epochs = epochs
        self.learning_rate = learning_rate
        regions_train, regions_test = train_test_split(get_regions_with_labels(), test_size=test_size)
        # Parameters for data generator
        self.params = {
            'dim': input_shape,
            'batch_size': batch_size,
            'n_classes': classes_number
        }
        # Generators
        self.training_generator = BlurRegionsDataGenerator(regions_train, **self.params)
        self.validation_generator = BlurRegionsDataGenerator(regions_test, **self.params)
        self.model = Sequential(
            [
                Conv2D(96, (7, 7), input_shape=input_shape),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.2),
                Conv2D(256, (5, 5)),
                Activation('relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.2),
                Flatten(),
                Dense(1024),
                Activation('relu'),
                Dropout(0.2),
                Dense(classes_number),
                Activation('softmax')
            ]
        )

        Adam(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        self.model.summary()
        self.history = None

    def get_model(self):
        return self.model

    def train_model(self):
        hist = self.model.fit(
            self.training_generator,
            epochs=self.epochs,
            validation_data=self.validation_generator,
            use_multiprocessing=True,
            workers=6,
            verbose=1
        )
        self.history = hist
        return hist

    def predict(self, image_path):
        mask = np.zeros((BLUR_ROWS, BLUR_COLS))
        regions = get_img_regions(image_path)
        crops = []
        for region in regions:
            crop = np.array([region.get_crop()])
            prediciton = np.argmax(self.model.predict(crop, batch_size=1))
            mask[region.row, region.col] = prediciton
        return mask

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def test(self):
        return self.model.evaluate(self.validation_generator)

    def save_model(self, dir_path):
        file = open(dir_path + "motionblur.h5", 'a')
        self.model.save(dir_path + "motionblur.h5")
        file.close()

    def load_model(self, filename):
        self.model = keras.models.load_model(filename)
