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
    # model.add(Activation("linear"))
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


def predict_lin_reg(x_train, y_train):
    reg = linear_model.LinearRegression()
    # x = np.array(x_train)
    # x = x.reshape((-1, 1))
    # y = np.array(y_train)
    reg.fit(x_train, y_train)
    return reg


# fig is matplotlib object
# x_train and y_train must be list
def draw_lin_reg(fig, x_train, y_train):
    x = np.array(x_train)
    x = x.reshape((-1, 1))

    mn = np.min(x_train)
    mx = np.max(x_train)

    y = np.array(y_train)
    reg = predict_lin_reg(x, y)

    p0 = [mn, mx]
    p1 = [mn * reg.coef_[0] + reg.intercept_, mx * reg.coef_[0] + reg.intercept_]

    plt.plot(p0, p1, figure=fig, color='cyan')
    plt.plot([mn, mx], [mn, mx], figure=fig, color='red')

    d = np.round(np.rad2deg(np.arctan(reg.coef_[0])), 2)
    print('regression line degree: ' + str(d))
    print('regression intercept: ', str(reg.intercept_))


def show_45_deg_stats(
        method_name,
        actual_data,
        pred_data):

    fig, ax = plt.subplots(1, 1)
    plt.scatter(
        pred_data,
        actual_data,
        marker='o',
        label='prediction',
        color='blue',
        figure=fig)

    plt.xlabel('prediction', figure=fig)
    plt.ylabel('actual', figure=fig)
    plt.title(method_name, figure=fig)

    # draw_lin_reg(
    #     fig,
    #     test_data,
    #     pred_data)

    x = np.array(pred_data)
    x = x.reshape((-1, 1))

    mn = np.min(pred_data)
    mx = np.max(pred_data)

    y = np.array(actual_data)
    reg = predict_lin_reg(x, y)
    r2 = reg.score(x, y)
    p0 = [mn, mx]
    p1 = [mn * reg.coef_[0] + reg.intercept_, mx * reg.coef_[0] + reg.intercept_]

    plt.plot(p0, p1, figure=fig, color='cyan')

    # mn = min(mn, np.min(actual_data))
    # mx = max(mx, np.max(actual_data))

    plt.plot([mn, mx], [mn, mx], figure=fig, color='red')

    d = np.round(np.rad2deg(np.arctan(reg.coef_[0])), 2)
    print('regression line degree(' + method_name + "): " + str(d))
    print('regression line degree(' + method_name + "): " + str(reg.intercept_))
    print('regression R2(' + method_name + "): " + str(r2))

    ax.legend(['data', 'regression line', '45 degree line'])
