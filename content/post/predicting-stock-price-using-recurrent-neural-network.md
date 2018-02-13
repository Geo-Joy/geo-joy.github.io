---
section: post
date: "2018-02-03"
title: "Predicting Stock Price using Recurrent Neural Network"
description: "A practical implementation of RNN in predicting the stock price."
slug: "predicting-stock-price-using-recurrent-neural-network"
tags:
 - practical
 - rnn
 - project
---

![predicting-stock-price-intro.jpg](/images/articles/2018/RNN/predicting-stock-price-intro.jpg "predicting-stock-price-intro.jpg")

# Prelude

``Brownian Motion`` stats that the future variations of the stock price are ``independent from the past``.

So it is actually ``impossible to predict`` the future stock price but one thing we can do or ``predict is the trend``.

So in the tutorial we will ``try to predict`` the upward and downward trend on ``Google Stock Price`` using ``LSTM``

# Going about

- We will try to get a better understand on the usage of LSTMs.
- This LSTM will be trained with ``6yrs`` of google stock price ``(2013-2017)``
- Predict the ``open of stock`` for first month of ``2018``
- NB: There is no ``SATURDAY`` and ``SUNDAY`` data in the dataset.

# Downloading the Dataset

### TrainSet
https://finance.yahoo.com/quote/GOOG/history?period1=1356978600&period2=1514658600&interval=1d&filter=history&frequency=1d

- 01 Jan 2013 to 31 Dec 2017

![google-stock-train.png](/images/articles/2018/RNN/google-stock-train.png "google-stock-train.png")
![google-stock-train.png](/images/articles/2018/RNN/google-stock-train-graph.png "google-stock-train.png")
### TestSet
https://finance.yahoo.com/quote/GOOG/history?period1=1514745000&period2=1517337000&interval=1d&filter=history&frequency=1d

- 01 Jan 2018 to 31 Jan 2018

![google-stock-test.png](/images/articles/2018/RNN/google-stock-test.png "google-stock-test.png")
![google-stock-test.png](/images/articles/2018/RNN/google-stock-test-graph.png "google-stock-test.png")

---

<br/>

# Data Preprocessing
```py
## Data Preprocessing
import numpy as np #allow us to do array manipulation
import matplotlib.pyplot as plt #to visualize the data
import pandas as pd #to import and manage the dataset
```
## Importing Data

To import the data from the excel we downloaded, we will use the pandas ``` read_csv ``` function.

read_csv imports data as a ``` DataFrame ``` 

#### Dataframe
https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

> A DataFrame is a two-dimentional labeled data structures with columns of potentially different types.<br/>
check out: https://www.youtube.com/watch?time_continue=44&v=CLoNO-XxNXU
 
```py
## Importing the training set
# importing as dataframe using pandas
dataset_train = pd.read_csv('Google_Stock_Price_Train_2013-2017.csv');
# Select the required column using iloc method and .values converts dataframe to array
# .iloc[all_columns: only 1 row (open stock)]
# convert the dataset to numpy array as neural network accepts only arrays as inputs
training_set = dataset_train.iloc[:,1:2].values # should give as range if we give [:,1] we just get a vector what we need is numpy array
```

![stock_dataframe.png](/images/articles/2018/RNN/stock_dataframe.png "stock_dataframe.png")
<br/>
![stock_training_set.png](/images/articles/2018/RNN/stock_training_set.png "stock_training_set.png")

## Feature Scaling

Feature Scaling can be done in two methods
1. Standardisation
2. Normalisation

Here we will use Normalisation as we are using a sigmoid function as an activation function in output layer.

For this we will use the MinMaxScalar class from the pre-processing module in scikit learn library.

```py
from sklearn.preprocessing import MinMaxScaler
```

![feature_scaling.png](/images/articles/2018/RNN/feature_scaling.png "feature_scaling.png")
<center>image borrowed from <a href="https://www.superdatascience.com">Super Data Science</a></center>

<br/>

```py
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # to get all stock price between 0 & 1
# apply this sc object on our data
training_set_scaled = sc.fit_transform(training_set) # fit_transform is used to convert all data between 0 & 1
```

![stock_scaled.png](/images/articles/2018/RNN/stock_scaled.png "stock_scaled.png")

## Creating DataStructure for the RNN

Here we tell the RNN to take 60 timesteps and predict 1 output.

That is at time ``t`` the RNN looks back ``60`` stock prices before time ``t`` and based on the trends its capture it trys to predict the output at ``t+1``

This is to prevent ``OVERFITTING`` and ``UNDERFITTING``

```py
## Creating data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, 1260): # 60 inputs till the total range of the dataset
    X_train.append(training_set_scaled[i-60:i, 0]) # 60-60 = 0 so 0 to 60 indexes ,0 is for the column(we have only 1 column now :P)
    y_train.append(training_set_scaled[i, 0]) # index starts at 0 :P
```

converting X_train and Y_train to numpy array sinse they are now a list

```py
X_train, y_train = np.array(X_train), np.array(y_train)
```

reshaping to be compatible to the neural network

```py
# Reshaping - to add additional dimentions
# (batch_size, timesteps, input_dim)
# batch_size : total number of stock prices #X_train.shape[0] = gets the total rows 
# timesteps which is 60 # X_train.shape[0] = gets the total columns
# input_dim = 1 since we are using only 1 indicator
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) #(batch_size, timesteps, input_dim)
```

# Building the RNN

```py
## Building the RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising th RNN
# Classification is for predicting a category or a cass
# Regression is for predicting a continuous value
regressor = Sequential()

# Adding the first LSTM layer and some dropout reqularisation (to avoid overfitting)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# for the dropout
regressor.add(Dropout(0.2)) # 20% dropout - neurons in LSTM will be ignored in each iteration of the training

#second layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#third layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#forth layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


## Adding the Output Layer
regressor.add(Dense(units = 1))


## Compiling the RNN with loss function
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

## Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
```

When we run the above code we start the training process. After completion we need to follow the same steps on the ``test dataset``

```py
## P3 - Making the prediction and visualising the results

# Getting the real stock price of 2018
dataset_test = pd.read_csv('Google_Stock_Price_Test_2018-2018.csv');
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price
#concatenating both the train and test sets
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0); # 1 for horizontal concatenation & 0 for vertical
# getting the new inputs fo each financial day
inputs = dataset_total[len(dataset_total)-len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
    
X_test = np.array(X_test)
# for the 3D structure
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price);
```

# Visualising the results

We use ``pyplot`` method from matplotlib for visualising

```py
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```

![google-stock-predicted.png](/images/articles/2018/RNN/google-stock-predicted.png "google-stock-predicted.png")

``We have successfully implemented our stock price predictor for 2018``

