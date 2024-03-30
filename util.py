import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
import cfg

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn import preprocessing


def make_lstm_model():
    model = Sequential()
    model.add(LSTM(
        units=20,
        input_dim=1,

        return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.1))

    model.add(Dense(
        units=1))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="rmsprop")
    return model


def make_model(window_size):
    model = Sequential()
    model.add(Dense(cfg.l1, input_dim=window_size, activation="tanh"))
    model.add(Dense(cfg.l2, activation="tanh"))
    model.add(Dense(cfg.l3, activation="tanh"))
    model.add(Dense(1, activation='sigmoid'))
    #model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(
        np.abs(
            percentage_error(
                np.asarray(y_true),
                np.asarray(y_pred)))) * 100


def read_data(csv_path):
    # loading data
    dt = pd.read_csv(csv_path)
    dt.head()

    ls_data = []

    # showing data
    print('total samples: ' + str(dt.shape[0]))
    for i in range(dt.shape[0]):
        s = dt.iloc[i, 4]  # .to_string().split(';')
        s = s.replace(',', '')
        number = float(s)
        ls_data.append(number)

    return ls_data


def mean_error(act, y_pred):
    # assuming y and y_pred are numpy arrays
    act = np.array(act)
    y_pred = np.array(y_pred)
    return np.mean(y_pred - act)


def draw_nn_perf(ann_history):
    # plt.plot(ann_history.history['accuracy'])
    # plt.plot(ann_history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    plt.plot(ann_history.history['loss'])
    plt.plot(ann_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def print_errors(
        predictor_name,
        actual_data,
        prediction_data):
    mse = mean_squared_error(
        actual_data,
        prediction_data)
    print(
        "{0} ({1}): {2:0.3f}".format(
        'MSE', predictor_name, mse))

    me = mean_error(
        actual_data,
        prediction_data)
    print(
        "{0} ({1}): {2:0.3f}".format(
        'ME', predictor_name, me))

    mae = mean_absolute_error(
        actual_data,
        prediction_data)
    print(
        "{0} ({1}): {2:0.3f}".format(
        'MAE', predictor_name, mae))

    mape = mean_absolute_percentage_error(
        actual_data,
        prediction_data)
    print(
        "{0} ({1}): {2:0.3f}".format(
        'MAPE', predictor_name, mape))


def predict_lin_reg(x_train, y_train, x_test):
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    return reg.predict(x_test)

