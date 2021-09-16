#discription this program uses artificial nueral network called long short term memory to predict nvda stock price using the past 60 days
import math
from typing import ValuesView
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocesssing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime as dt

company = 'FB'
start = dt.datetime(2010,1,1)
end = dt.datetime(2020,1,1)

data = web.DataReader(company, 'yahoo', start, end)

#prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_tranformed(data['close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-predication_days:x, 0])
    y_train.append(scaled_data[x,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shaped[0], x_train.shape[1],1))

#model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(unit=1))#prediction of the next closing 

model.complie(optimzer='adam', loss='mean_squaredd_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

''' Test the model accuracy on existing data'''

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
acutal_prices = test_data['clsoe'].values

total_dataset = pd.concat((data['close'], test_data['close']), axis = 0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days].values
model_inputs = model_input.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#make predictions

x_test = []

for x in range(prediciton_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediciton_days:x,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_tranform(predicted_prices)


plt.plot(actual_prices, color="black", label=f"actual {company} price")
plt.plot(predicted_prices, color='greem', label=f"predicted {company} price")
plt.title(f"{company} share price")
plt.xlabel('time')
plt.ylabel(f'{company} share price')
plt.legend()
plt.show()

#predicting tommorow
real_data = [model_inputs[len(model_inputs)+ 1 - prediction_days:len(model_input + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data[0], real_data.shape[1],1))

prediciton = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"prediciton:{prediction}")




