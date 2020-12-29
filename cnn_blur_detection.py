from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam


class MotionBlurDetectionCNN:

    def __init__(self, input_shape=(30, 30, 3), classes_number=2, epochs=100, learning_rate=0.01):
        self.input_shape = input_shape
        self.classes_number = classes_number
        self.epochs = epochs
        self.learning_rate = learning_rate
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
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])

        self.model.summary()

    def get_model(self):
        return self.model

    def train_model(self, X_train, y_train):
        hist = self.model.fit(X_train, y_train, epochs=self.epochs)
        return hist

    def predict(self, test_image):
        return self.model.predict(test_image)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def save_model(self, dir_path):
        file = open(dir_path + "motionblur.h5", 'a')
        self.model.save(dir_path + "motionblur.h5")
        file.close()
