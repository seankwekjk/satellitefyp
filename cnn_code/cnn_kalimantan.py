from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


def train(x_train, y_train, x_test, y_test):
    model = Sequential()

    # add model layers
    model.add(Conv2D(32, kernel_size=50, strides=(5, 5), activation="relu", input_shape=(756, 1211)))
    model.add(Conv2D(64, kernel_size=10, strides=(5, 5), activation="relu"))
    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))

    # compile model using accuracy to measure model performance
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
