import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from sklearn.metrics import mean_squared_error as mse

from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from sklearn import preprocessing
#from keras.wrappers.scikit_learn import KerasRegressor
#from keras.layers.recurrent import LSTM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cfg
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn import preprocessing
from statsmodels.tsa.stattools import acf
from keras.layers import LSTM

def read_data(csv_path):
    # loading data
    dt = pd.read_csv(csv_path)
    dt.head()

    ls_data = []

    # showing data
    print('total samples: ' +str(dt.shape[0]))
    for i in range(dt.shape[0]):
        s = dt.iloc[i, 4] #.to_string().split(';')
        s = s.replace(',', '')
        number = float(s)
        ls_data.append(number)

    return ls_data

def mean_error(y, y_pred):
    # assuming y and y_pred are numpy arrays
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.mean(y_pred - y)


# loading data
ls_data = read_data(cfg.csv_path)

# showing data
plt.title('raw data')
plt.plot(ls_data, marker=1)
plt.show()

#split into test and train
trn_count = int(len(ls_data) * cfg.trn_sz)
train, test = ls_data[0:trn_count], ls_data[trn_count:len(ls_data)]

print('# sample train: ' + str(trn_count))
print('# sample test: ' + str(len(test)))

model = ARIMA(train, order=cfg.ARIMA_order)
model_fit = model.fit()
# acf_1 = acf(ls_data)
# plt.plot(acf_1)
# test_df = pd.DataFrame([acf_1]).T
# test_df.columns = ["Autocorrelation"]
# test_df.index += 1
# test_df.plot(kind='bar')
# pacf_1 = pacf(ls_data)
# plt.plot(pacf_1)
# plt.show()
# test_df = pd.DataFrame([pacf_1]).T

"""
Arima Rolling Forecast
"""
predicted1, resid_test = [], []
history = train
for t in range(len(test)):
    model = ARIMA(history, order=(5, 0, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    resid_test.append(test[t] - output[0])
    predicted1.append(yhat)
    obs = test[t]
    history.append(obs)
    print(str(t) + ' - ' + 'predicted=%f, expected=%f' % (yhat, obs))

test_resid = []
for i in resid_test:
     test_resid.append(i)
mse = mean_squared_error(test_resid, predicted1, squared=False)
print('Test MSE (ARIMA): %.3f' % mse)

me = mean_error(test_resid, predicted1)
print('Test ME (ARIMA): %.3f' % me)

mae = mean_absolute_error(test_resid, predicted1)
print('Test MAE (ARIMA): %.3f' % mae)

plt.title('ARIMA')
plt.plot(test, label='Test', color='blue', marker=1)
plt.plot(predicted1, label='Prediction', color='red', marker=2)
plt.legend()
plt.show()

"""
Residual Diagnostics
"""
train, test = ls_data[0:trn_count], ls_data[trn_count:len(ls_data)]
model = ARIMA(train, order=(1, 0, 0))
model_fit = model.fit()
print(model_fit.summary())
# plot residual errors
more_info = pd.DataFrame(model_fit.resid)
more_info.plot()
plt.show()
more_info.plot(kind='kde')
plt.show()
print(more_info.describe())
#plot the acf for the residuals
acf_1 = acf(model_fit.resid)[1:20]
plt.plot(acf_1)
test_df = pd.DataFrame([acf_1]).T
test_df.columns = ["Autocorrelation"]
test_df.index += 1
test_df.plot(kind='bar')
plt.show()
"""
Hybrid Model
"""
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
    model.add(Dense(7, input_dim=window_size, activation="tanh"))
    model.add(Dense(7, activation="tanh"))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if(cfg.model_type == 'lstm'):
    model = make_lstm_model()
else:
    model = make_model(cfg.win_sz)

mms = preprocessing.MinMaxScaler()
train = np.array(train).reshape(-1, 1)
#todo may test must be used here
train_scaled = mms.fit_transform(train)

train_x, train_y = [], []
for i in range(0, len(train_scaled) - cfg.win_sz):
    train_x.append(train_scaled[i:i + cfg.win_sz])
    train_y.append(train_scaled[i + cfg.win_sz])

new_train_X, new_train_Y = [], []
for i in train_x:
    new_train_X.append(i.reshape(-1))
for i in train_y:
    new_train_Y.append(i.reshape(-1))
new_train_X = np.array(new_train_X)
new_train_Y = np.array(new_train_Y)
model.fit(new_train_X, new_train_Y, epochs=500, batch_size=512, validation_split=.05, verbose=0)

test_extended = train.tolist()[-1*cfg.win_sz:] + test_resid
test_data = []
for i in test_extended:
    try:
        test_data.append(i[0])
    except:
        test_data.append(i)
test_data = np.array(test_data).reshape(-1,1)

# min_max_scaler
mms = preprocessing.MinMaxScaler()
test_scaled = mms.fit_transform(test_data)
test_X, test_Y = [], []
for i in range(0 , len(test_scaled) - cfg.win_sz):
    test_X.append(test_scaled[i:i+cfg.win_sz])
    test_Y.append(test_scaled[i+cfg.win_sz])
    new_test_X,new_test_Y = [], []
for i in test_X:
    new_test_X.append(i.reshape(-1))
for i in test_Y:
    new_test_Y.append(i.reshape(-1))
new_test_X = np.array(new_test_X)
new_test_Y = np.array(new_test_Y)

predictions = model.predict(new_test_X)
predictions_rescaled=mms.inverse_transform(predictions)
y = pd.DataFrame(new_train_Y)
pred = pd.DataFrame(predictions)
plt.plot(y[0: len(pred)], color='blue', label='actual')
plt.plot(pred, color='red', label='prediction')
plt.title('ANN')
plt.show()

mse = mean_squared_error(test, predictions_rescaled)
print('Test MSE (ANN): %.3f' % mse)

me = mean_error(test, predictions_rescaled)
print('Test ME (ANN): %.3f' % me)

mae = mean_absolute_error(test, predictions_rescaled)
print('Test ME (ANN): %.3f' % mae)

predictions_rescaled = predictions_rescaled.squeeze()
pred_final = predictions_rescaled + predicted1

mse = mean_squared_error(test, pred_final)
print('Test MSE (Hybrid): %.3f' % mse)

me = mean_error(test, pred_final)
print('Test ME (Hybrid): %.3f' % me)

mae = mean_absolute_error(test, pred_final)
print('Test MAE (Hybrid): %.3f' % mae)

y = pd.DataFrame(test)
pred = pd.DataFrame(pred_final)
plt.plot(y)
plt.plot(pred, color='r')
plt.title('Hybrid')
#p.plot()
plt.show()
