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
        model.add(Conv2D(32, kernel_size=2, strides=(2, 2), activation="relu", input_shape=(768, 1023, 1)))
        # model.add(Conv2D(64, kernel_size=3, strides=(3, 3), activation="relu"))
        model.add(Flatten())
        # model.add(Flatten())
        model.add(Dense(10, activation="relu"))
        model.add(Dense(1))

        # compile model using accuracy to measure model performance
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    estimator = KerasRegressor(build_fn=build_model, epochs=20, batch_size=5, verbose=0)
    estimator.fit(x_train, y_train)

    predictions = estimator.predict(x_test)
    print('prediction')
    for prediction in predictions:
        print(prediction)
    print('val')
    for val in y_test:
        print(val)

    '''
    # train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
    '''
