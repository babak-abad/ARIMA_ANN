from pandas import read_csv
import pandas as pd
import cfg
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
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

series = read_data(cfg.csv_path)
series = series[0:300]
#series.index = series.index.to_period('M')
# split into train and test sets
X = series
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
 model = ARIMA(history, order=(5,1,0))
 model_fit = model.fit()
 output = model_fit.forecast()
 yhat = output[0]
 predictions.append(yhat)
 obs = test[t]
 history.append(obs)
 print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()