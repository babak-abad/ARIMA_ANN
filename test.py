# predictor_name = 'ANN'
# mse = 0.123456
# str = "{0} ({1}): {2:0.3f}".format(
#     'MSE', predictor_name, mse)
# print(str)
import numpy as np

x = np.array([1, 2, 3])
x = x.reshape((-1, 1))

from sklearn.linear_model import LinearRegression
x = np.array([[1], [2], [3], [4]])
# y = 1 * x_0 + 2 * x_1 + 3
y = 2*x + 5
reg = LinearRegression().fit(x, y)
reg.score(x, y)
#1.0
reg.coef_
#array([1., 2.])
#reg.intercept_
#3.0...
print(reg.predict(np.array([[3, 5]])))
#array([16.])