import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
import seaborn as sns




df = pd.read_csv('/Users/spusegao/Downloads/GE.csv')

# Separate Dates from the rest of the data
train_dates = pd.to_datetime(df['Date'])

# Variables for Training 
cols = list(df)[1:6]

df_for_training = df[cols].astype(float)

#df_for_plot = df_for_training.tail(5000)
#df_for_plot.plot.line()

# LSTM uses sigmoid and tanh activations that are sensitive to the magnitude 
# Normalize the data 

scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


trainX = []
trainY = []

n_future = 1 
n_past = 14


for i in range(n_past, len(df_for_training_scaled)-n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future -1:i + n_future,0])
    
trainX, trainY = np.array(trainX), np.array(trainY)

print("Shape of training set == {}.".format(trainX.shape))
print("Shape of test set == {}.".format(trainX.shape))


# DEFINE THE LAYERS
    
model=Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1],trainX.shape[2]), return_sequences=True))
model.add(LSTM(32,  activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))
model.compile(loss='mse', optimizer='adam')
model.summary()


# fit model

history = model.fit(trainX, trainY,epochs=10, batch_size=16, validation_split=0.1, verbose=0.1)




#Forecasting...
#Start with the last day in training date and predict future...

#plt.plot(history.history['loss'], label='Training Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
#plt.legend()

n_future_new=90
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future_new, freq='1d').tolist()

forecast = model.predict(trainX[-n_future_new:])

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform

forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]

forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())
    
df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])


original = df[['Date', 'Open']]
original['Date']=pd.to_datetime(original['Date'])
original = original.loc[original['Date'] >= '2020-5-1']

sns.lineplot(original['Date'], original['Open'])
sns.lineplot(df_forecast['Date'], df_forecast['Open'])
