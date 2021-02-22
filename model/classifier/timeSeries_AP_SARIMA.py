import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import ADFTest

plt.style.use('dark_background')

df = pd.read_csv('/Users/spusegao/Downloads/AirPassengers.csv')
print(df.dtypes)

df['Month'] = pd.to_datetime(df['Month'])
print(df.dtypes)

df.set_index('Month',inplace=True)

plt.plot(df['Passengers'])

# Is the data stationary? 

adf_test=ADFTest(alpha=0.05)
adf_test.should_diff(df)

# Dickey-Fuller test

adf, pvalue, usedlag_, nobs_, critical_calues_, icbest_  = adfuller(df)

print("pvalue=", pvalue)
# pvalue= 0.991880243437641 => Means the data is not stationary
# since the data is not stationary we may need to use SARIMA and not ARIMA

#df['year'] = [d.year for d in df.index]
#df['month'] = [d.strftime('%b') for d in df.index]
#years = df['year'].unique

# plot monthly and yearly values as box plot
#sns.boxplot(x='year', y='Passengers',data=df)
#sns.boxplot(x='month', y='Passengers',data=df)
#print(years)

# extract and plot trend m seasonal and residuals

from statsmodels.tsa.seasonal import seasonal_decompose

decomposed = seasonal_decompose(df['Passengers'],
                                model='additive')


# Additive Time series :
# Values = Base LEvel + Trend + Seasonality + Error
# Multivariate Time Series : 
# Values = Base Level * Trend * * Error

trend = decomposed.trend
seasonal = decomposed.seasonal
residual = decomposed.resid

# Lets see these values in a plot...

plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['Passengers'], label='Original', color='yellow')
plt.legend(loc='upper left')

plt.subplot(412)
plt.plot(trend, label='Original', color='yellow')
plt.legend(loc='upper left')

plt.subplot(413)
plt.plot(seasonal, label='Original', color='yellow')
plt.legend(loc='upper left')

plt.subplot(414)
plt.plot(residual, label='Original', color='yellow')
plt.legend(loc='upper left')

plt.show()

# Auto ARIMA suggests best parameters and model based on 
# AIC metric (relative quality of statustical models)

from pmdarima.arima import auto_arima

# AutoArima gives us the best model suited for the data
# p - number of autoregressive terms (AR)
# q - Number of moving averages terms (MA)
# d - Number of non-seasonal differences
# 
# p,d,q represents non-seasonal components
# P, D, Q represent seasonal components

arima_model =  auto_arima(df['Passengers'], start_p = 1, d = 1, start_q = 1, 
                          max_p = 5, max_q = 5, max_d = 5, m = 12, 
                          start_P = 0, D=1, start_Q=0, max_P=5, max_D=5, max_Q=5,
                          seasonal=True,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True,
                          n_fits=50)

# Print the summary
print(arima_model.summary())

# Note down the model and details
# Model:             SARIMAX(0, 1, 1)x(2, 1, [], 12) 

# split the data into train and test
# This is not a random selection....assign from 0-66% to train
size=int(len(df)*0.66)
X_train, X_test = df[0:size], df[size:len(df)]

# fit the model on the training set

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(X_train['Passengers'],
                order = (0,1,1),
                seasonal_order=(2,1,1,12))


result = model.fit()
result.summary()

# Train prediction
start_index=0
end_index=len(X_train)-1
train_prediction = result.predict(start_index, end_index)

# Lets Predict now
start_index=len(X_train)
end_index=len(df)-1
prediction = result.predict(start_index, end_index).rename('Predicted Passengers')

# Plot the prediction and actual Values
prediction.plot(legend=True)
X_test['Passengers'].plot(legend=True)

import math
from sklearn.metrics import mean_squared_error

trainScore = math.sqrt(mean_squared_error(X_train, train_prediction))
print('Train Score : %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(X_test, prediction))
print('Train Score : %.2f RMSE' % (testScore))

from sklearn.metrics import r2_score

score= r2_score(X_test, prediction)



forecast = result.predict(start = len(df),
                          end = (len(df)-1) + 3 * 12,
                          typ = 'levels').rename('Forecast')

plt.figure(figsize=(12,8))
plt.plot(X_train, label='Training', color='green')
plt.plot(X_test, label='Test', color='red')
plt.plot(forecast, label='Forecast', color='yellow')
plt.legend(loc='left corner')
plt.show()



# AUTOCORRELATION
# Autocorrelation is the correlation of a series with its own lags.
# Plot lag on x axis and correlation on y asix
# Any correlation above the confidence lines are statistically significant

from statsmodels.tsa.stattools import acf

acf_144 = acf(df.Passengers, nlags=144)
plt.plot(acf_144)

# Obtaining the same with single line command but with more information
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Passengers)

# Horizontal bands provide a 95% and 99%(dashed) confidence level bands
# In this case a strong positive correlation for lags below 40 months                     
              