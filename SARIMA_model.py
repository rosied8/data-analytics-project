#the package installation
# scipy
import scipy
# numpy
import numpy
# matplotlib
import matplotlib
# pandas
import pandas
# statsmodels
import statsmodels
# scikit-learn
import sklearn
#data processing and data loaing
import pandas as pd
file=pd.read_csv('rawData.csv')
import pandas as pd
from datetime import datetime
def parse(x):
    return datetime.strptime(x,"%Y %m %d %H")
dataset=pd.read_csv("rawData.csv",parse_dates = [['year', 'month', 'day', 'hour']],index_col=0,date_parser=parse)
dataset.head()
dataset.drop("No",axis=1,inplace=True)
dataset.head()
dataset.index.name="date"
dataset.columns=['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.head()
dataset.fillna(0,inplace=True)
dataset.head()
dataset=dataset[24:]
dataset= dataset.resample('D').mean().to_period('D')
#divide the data into train set,test set and validation set
import matplotlib.pyplot as plt
test_index=y.shape[0]
train_len=int((y.shape[0])*0.7)
train=dataset.iloc[0:train_len]
validation=dataset.iloc[train_len:test_index]
y_train = dataset["pollution"]
y_val=dataset["pollution"]
#feature enginerring of the data
from statsmodels.tsa.seasonal import seasonal_decompose
#y_train.index=y_train.index.to_timestamp()
result = seasonal_decompose(y_train, model="additive")
result.plot()
plt.show()
y_train["2010-01-02":"2010-07-01"].plot(figsize=(15, 6))
plt.show()
#use adfuller to check the stability
from statsmodels.tsa.stattools import adfuller
#if the time series is stationary or not
result = adfuller(y_train)
print(result)
if result[1]<0.05:
    print(True)
# SARIMA modelling -parameter choose
import itertools
order_d = 0
p = q = range(0, 3)
# Generate all different combinations of p, q and q triplets
pdqlist = list(itertools.product(p, q))
pdq = [(x[0], order_d, x[1]) for x in list(itertools.product(p, q))]
# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], 1, x[1], 7) for x in list(itertools.product(p, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
import warnings
import statsmodels.api as sm
warnings.filterwarnings("ignore") # specify to ignore warning messages
maxr = 100000
param_best = 0
param_seasonal_best = 0

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_train,
                                           order=param,
                                           seasonal_order=param_seasonal,
                                           enforce_stationarity=True,
                                           enforce_invertibility=False)
            results = mod.fit()
            if results.aic < maxr:
                maxr = results.aic
                param_best = param
                param_seasonal_best = param_seasonal

            print('SARIMA{}x{}- AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
print(param)
#check the best parameter
print(param_best)
print(param_seasonal_best)
#refit the model with best parameter
mod = sm.tsa.statespace.SARIMAX(y_train,
                                order=(1, 0, 1),
                                seasonal_order=(0, 1, 1, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])
#confirm the precise of result
results.plot_diagnostics(figsize=(15, 12))
plt.show()

