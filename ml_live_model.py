# -*- coding: utf-8 -*-
"""ML_Live_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bWfUzwqvOCN7Wd3DCbXSPA2FpRBjrifP
"""

import tensorflow as tf
import numpy as np
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Replace YOUR_API_KEY with your Alpha Vantage API key
ts = TimeSeries(key='WO6IIEBNMUTO92UN', output_format='pandas')
data, meta_data = ts.get_daily_adjusted(symbol='IBM', outputsize='full')

# Extract the 'Adj Close' column from the data
data = data['4. close'].values.reshape(-1, 1)

# Normalize the data

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Create input data and labels


def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i+look_back, 0])
    return np.array(X), np.array(Y)

look_back = 7


# Number of previous days to use as input features
X, Y = create_dataset(data, look_back)

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
trainX, testX = X[0:train_size, :], X[train_size:len(X), :]
trainY, testY = Y[0:train_size], Y[train_size:len(Y)]

# Reshape the input data to be 3-dimensional
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Create the LSTM model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=50, return_sequences=True,
                               input_shape=(look_back, 1)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=50))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
history = model.fit(trainX, trainY, epochs=50,
                    batch_size=32, validation_data=(testX, testY))

testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


plt.figure(figsize=(18, 9))
plt.plot(testY[0], label='Actual')
plt.plot(testPredict[:, 0], label='Predicted')
plt.legend()
plt.show()

model.save('model.h5')
