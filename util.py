import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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