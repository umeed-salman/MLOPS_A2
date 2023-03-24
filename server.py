from flask import Flask, render_template, request
from alpha_vantage.timeseries import TimeSeries
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


app = Flask(__name__)


# Replace YOUR_API_KEY with
# your Alpha Vantage API key
ts = TimeSeries(key='WO6IIEBNMUTO92UN',
                output_format='pandas')

# Load the LSTM model
model = tf.keras.models.load_model('model.h5')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))


# Function to create the input data and labels
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i+look_back, 0])
    return np.array(X), np.array(Y)


# Home page route
@app.route('/')
def home():
    # Retrieve the stock data
    symbol = 'IBM'
    data, meta_data = ts.get_daily_adjusted(symbol = symbol,
                                            outputsize ='compact')
    dates = []
    for i in data.index:
        dates.append(i.strftime("%m/%d/%Y"))

    # Extract the 'Adj Close' column from the data
    data = data['4. close'].values.reshape(-1, 1)

    # Normalize the data
    data = scaler.fit_transform(data)

    # Create input data and labels
    
    
    look_back = 7 # Number of previous days to use as input features
    X, Y = create_dataset(data, look_back)

    # Reshape the input data to be 3-dimensional
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Make predictions using the LSTM model
    predicted = model.predict(X)
    predicted = scaler.inverse_transform(predicted)

    # Get the last 7 days of actual stock prices
    # and the predicted price for the next day
    # last_7_days = scaler.inverse_transform(data[-7:])

    last_7_days = []
    for i in range(-7, 0):
        last_7_days.append([dates[i],
                            scaler.inverse_transform([data[i]])[0][0]])

    next_day_predicted = predicted[-1][0]

    # Render the template and pass the data to it
    return render_template('home.html', symbol=symbol,
                           last_7_days=last_7_days,
                           next_day_predicted=next_day_predicted)


# Prediction page route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Get the user input
        user_input = np.array(
            request.form.getlist('user_input'),
            dtype=np.float32)

        # Normalize the user input
        user_input = scaler.transform(user_input.reshape(-1, 1))

        # Reshape the user input to be compatible with the LSTM model
        user_input = np.reshape(user_input, (1, 7, 1))

        # Make predictions using the LSTM model
        predicted = model.predict(user_input)
        predicted = scaler.inverse_transform(predicted)

        # Get the predicted price for the next day
        next_day_predicted = predicted[0][0]

        # Render the template and pass the data to it
        return render_template('prediction.html',
                               user_input=request.form.getlist('user_input'),
                               next_day_predicted=next_day_predicted)

    # Render the template without any data
    # if the user hasn't submitted the form yet
    return render_template('prediction.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
