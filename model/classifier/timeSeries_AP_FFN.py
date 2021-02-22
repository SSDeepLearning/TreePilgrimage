import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import mean_squared_error



df = pd.read_csv('/Users/spusegao/Downloads/AirPassengers.csv', usecols=[1])
plt.plot(df)

# convert dataframe into numpy arrays
dataset=df.values
dataset=dataset.astype('float32')  # convert the values to float32 datatype

# Normalize the dataset as in NN the activation functions are sensitive to the magnetude of numbers.

scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(dataset)

# split the data in train and test.
# we can not do this in a random way as the sequence of events is important for the time series
# Lets take the first 66% values for training and the remaining for testing..

train_size = int(len(dataset) * 0.66)
test_size = len(dataset)-train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# 
#
#



def to_sequences(dataset, seq_size=1):
    x = []
    y = []
    
    for i in range(len(dataset)-seq_size-1):
        window = dataset[i:(i+seq_size),0]
        x.append(window)
        y.append(dataset[i+seq_size,0])
        
    return np.array(x), np.array(y)


seq_size = 5

trainX, trainY  = to_sequences(train,seq_size)
testX, testY  = to_sequences(test,seq_size)

print("Shape of training set".format(trainX.shape))
print("Shape of test set".format(trainX.shape))


# Build the network

model=Sequential()
model.add(Dense(64, input_dim=seq_size, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
print(model.summary())


model.fit(trainX, trainY, validation_data=(testX,testY),
          verbose=2, epochs=100)


# Make predictions 

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# Run the inverse transformation on the prediction output.

trainPredict = scaler.inverse_transform(trainPredict)
trainY_inverse = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_inverse = scaler.inverse_transform([testY])


# Check model performance 
trainScore = math.sqrt(mean_squared_error(trainY_inverse[0],trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY_inverse[0],testPredict[:,0]))
print('Train Score: %.2f RMSE '%(testScore))

# shift train predictions for plotting
#we must shift the predictions so that they align on the x-axis with the original dataset. 
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[seq_size:len(trainPredict)+seq_size, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(seq_size*2)+1:len(dataset)-1, :] = testPredict
#testPredictPlot[len(train)+(seq_size)-1:len(dataset)-1, :] = testPredict


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
