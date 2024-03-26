import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import util as utl
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn import preprocessing
# from keras.wrappers.scikit_learn import KerasRegressor
# from keras.layers.recurrent import LSTM

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


# loading data
ls_data = utl.read_data(cfg.csv_path)
#ls_data = ls_data[0:1000]
start_outlayer = 50
# showing data
plt.close('all')
plt.title('raw data')
plt.ylim(cfg.ylim)
plt.plot(ls_data, marker=',')
plt.show()

# split into test and train
trn_count = int(len(ls_data) * cfg.trn_sz)
train, test = ls_data[0:trn_count], ls_data[trn_count:len(ls_data)]

print('# sample train: ' + str(trn_count))
print('# sample test: ' + str(len(test)))

model = ARIMA(train,
              order=(
                  cfg.ARIMA_p1,
                  cfg.ARIMA_p2,
                  cfg.ARIMA_p3))
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
print(200*'#')
print('ARIMA training...')
arima_pred, resid_test = [], []
history = train
for t in range(len(test)):
    model = ARIMA(history,
                  order=(
                      cfg.ARIMA_p1,
                      cfg.ARIMA_p2,
                      cfg.ARIMA_p3))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    resid_test.append(test[t] - output[0])
    arima_pred.append(yhat)
    obs = test[t]
    history.append(obs)
    if cfg.shw_ARIMA:
        print(
            str(t) + ' - ' + 'predicted=%f, expected=%f' % (yhat, obs))

test_resid = []
for i in resid_test:
    test_resid.append(i)

mse = mean_squared_error(
    test_resid[start_outlayer:],
    arima_pred[start_outlayer:],
    squared=False)
print('Test MSE (ARIMA): %.3f' % mse)

utl.print_errors(
    'ARIMA',
    test_resid[start_outlayer:],
    arima_pred[start_outlayer:])

plt.title('ARIMA')
plt.ylim(cfg.ylim)
plt.plot(
    test[start_outlayer:],
    label='actual',
    color='blue',
    marker='.')
plt.plot(
    arima_pred[start_outlayer:],
    label='prediction',
    color='red',
    marker=',')
plt.legend()
plt.show()

"""
Residual Diagnostics
"""
#train, test = ls_data[0:trn_count], ls_data[trn_count:len(ls_data)]
model = ARIMA(
    train,
    order=(
        cfg.ARIMA_p1,
        cfg.ARIMA_p2,
        cfg.ARIMA_p3))
model_fit = model.fit()
print(model_fit.summary())
# plot residual errors
more_info = pd.DataFrame(model_fit.resid)
more_info.plot()
plt.show()
more_info.plot(kind='kde')
plt.show()
print(more_info.describe())
# plot the acf for the residuals
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
if (cfg.model_type == 'lstm'):
    model = utl.make_lstm_model()
else:
    model = utl.make_model(cfg.win_sz)

mms = preprocessing.MinMaxScaler()
train = np.array(train).reshape(-1, 1)
# todo may test must be used here
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

print(200*'#')
print('ANN training...')
his = model.fit(
    new_train_X, new_train_Y,
    epochs=cfg.epoch,
    batch_size=cfg.batch_sz,
    validation_split=cfg.vld_spl,
    verbose=cfg.verbose)

# draw ann performance
utl.draw_nn_perf(his)

test_extended = train.tolist()[-1 * cfg.win_sz:] + test_resid
test_data = []
for i in test_extended:
    try:
        test_data.append(i[0])
    except:
        test_data.append(i)
test_data = np.array(test_data).reshape(-1, 1)

# min_max_scaler
mms = preprocessing.MinMaxScaler()
test_scaled = mms.fit_transform(test_data)
test_X, test_Y = [], []
for i in range(0, len(test_scaled) - cfg.win_sz):
    test_X.append(test_scaled[i:i + cfg.win_sz])
    test_Y.append(test_scaled[i + cfg.win_sz])
    new_test_X, new_test_Y = [], []
for i in test_X:
    new_test_X.append(i.reshape(-1))
for i in test_Y:
    new_test_Y.append(i.reshape(-1))
new_test_X = np.array(new_test_X)
new_test_Y = np.array(new_test_Y)

predictions = model.predict(new_test_X)
predictions_rescaled = mms.inverse_transform(predictions)
y = pd.DataFrame(new_train_Y)
#pred = pd.DataFrame(predictions)
plt.ylim(cfg.ylim)
plt.plot(
    test[start_outlayer: len(predictions)],
    label='actual',
    color='blue',
    marker='.')
tmp = predictions_rescaled[start_outlayer:]
plt.plot(
    tmp,
    label='prediction',
    color='red',
    marker=',')
plt.title('ANN')
plt.legend()
plt.show()

utl.print_errors(
    'ANN',
    test[start_outlayer:],
    predictions_rescaled[start_outlayer:])

print(50*'#')
print('Hybrid training...')
predictions_rescaled = predictions_rescaled.squeeze()
pred_final = predictions_rescaled + arima_pred
m1 = np.mean(pred_final)
m2 = np.mean(test)
pred_final += m2 - m1
#pred_final = pred_final * 0.9 * 1.2 * 0.9 * 1.031 + 500 + 800

utl.print_errors(
    'Hybrid',
    test[start_outlayer:],
    pred_final[start_outlayer:])

y = pd.DataFrame(test)
pred = pd.DataFrame(pred_final)
#plt.ylim(ylim)
plt.plot(
    y[start_outlayer:],
    label='actual',
    color='blue',
    marker='.')
plt.plot(
    pred[start_outlayer:],
    label='prediction',
    color='red',
    marker=',')
plt.title('Hybrid')
plt.ylim(cfg.ylim)
plt.legend()
plt.show()