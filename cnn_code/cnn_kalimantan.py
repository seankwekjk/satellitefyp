from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasRegressor


def train(x_train, y_train, x_test, y_test):
    # y_train = to_categorical(y_train)

    def build_model():
        model = Sequential()

        # add model layers
        model.add(Conv2D(32, kernel_size=10, strides=(5, 5), activation="relu", input_shape=(756, 1211, 1)))
        # model.add(Conv2D(64, kernel_size=10, strides=(5, 5), activation="relu"))
        model.add(Flatten())
        # model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dense(1))

        # compile model using accuracy to measure model performance
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    estimator = KerasRegressor(build_fn=build_model, epochs=20, batch_size=5, verbose=0)
    estimator.fit(x_train, y_train)

    predictions = estimator.predict(x_test)
    for x, prediction in enumerate(predictions):
        print(y_test[x] + ' vs ' + prediction)

    '''
    # train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
    '''
