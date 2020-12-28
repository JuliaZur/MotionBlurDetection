from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam

input_shape = (30, 30, 3)
classes_number = 2

model = Sequential(
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

epochs = 100
learning_rate = 0.01
decay = learning_rate / epochs
adam = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])

model.summary()


