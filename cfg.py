#network architecture
l1 = 20
l2 = 10
l3 = 5 # this must be 1 !

#network training
trn_sz = 0.5
epoch = 5000
batch_sz = 128
vld_spl = 0.5
model_type = 'other'
verbose = 0
csv_path = './data/all.csv'
win_sz = 50

#ARIMA hyper parameters
ARIMA_p1 = 1
ARIMA_p2 = 0
ARIMA_p3 = 0

#visualization
ylim = (0, 5000)
shw_ARIMA = False