# Stock Price Prediction using Hybrid BiGRU-LSTM in PyTorch

## Overview

This project implements a deep learning model to predict stock prices (Open, High, Low, Close) using historical data. It utilizes a hybrid neural network architecture combining a Bidirectional Gated Recurrent Unit (BiGRU) layer followed by multiple Long Short-Term Memory (LSTM) layers. The model is trained on historical stock data along with several calculated technical indicators to potentially improve prediction accuracy.

The script performs the following main steps:
1.  Loads historical stock data from a CSV file.
2.  Preprocesses the data (handles dates, checks for missing values/duplicates).
3.  Generates common technical indicators as additional features.
4.  Visualizes the raw data and technical indicators.
5.  Scales the features using `MinMaxScaler`.
6.  Transforms the data into sequences suitable for RNN training.
7.  Splits the data into training and validation sets.
8.  Defines and initializes the Hybrid BiGRU-LSTM model, loss function (MSE), and optimizer (Adam).
9.  Trains the model, monitoring training and validation loss.
10. Plots the training/validation loss curves.
11. Evaluates the model on training and validation data.
12. Plots the true vs. predicted values for a segment of the data around the train/validation split.
13. Outputs the predictions for the training and validation sets as pandas DataFrames.

## Features

* **Data Loading:** Reads stock data from a CSV file.
* **Preprocessing:** Converts date columns, sets date as index, converts columns to lowercase.
* **Feature Engineering:** Calculates technical indicators to augment the input features.
* **Data Scaling:** Normalizes features to the [0, 1] range.
* **Sequence Generation:** Creates input sequences and corresponding target values for time-series forecasting.
* **Hybrid Model:** Implements a `Hybrid BiGRU-LSTM` model combining BiGRU and LSTM layers with Dropout.
* **Training:** Trains the model using PyTorch, including a validation loop and progress printing.
* **Visualization:**
    * Plots raw Open, high, low, close (OHLC) data over time.
    * Plots calculated technical indicators.
    * Plots training and validation loss curves.
    * Plots true vs. predicted values for both training and validation sets.
* **Evaluation:** Generates predictions on training and validation sets.
* **Output:** Provides predictions as pandas DataFrames with corresponding dates.

## Model Architecture

The `Hybrid BiGRU-LSTM` model consists of:
1.  **Bidirectional GRU Layer:** Processes the input sequence in both forward and backward directions to capture past and future context.
2.  **Three Stacked LSTM Layers:** Process the output of the BiGRU layer sequentially, allowing the model to learn complex temporal dependencies at different scales.
3.  **Dropout Layers:** Applied after each LSTM layer to prevent overfitting during training.
4.  **Linear Layer:** A fully connected layer that maps the final LSTM hidden state to the desired output dimension (predicting all input features for the next time step).

## Technical Indicators Used

The `get_technical_indicators` function calculates the following features based on the 'close' price:
* **Moving Averages (MA):** 7-day (`ma7`) and 21-day (`ma21`).
* **Moving Average Convergence Divergence (MACD):** Calculated using 12-day and 26-day Exponential Moving Averages (EMAs). Intermediate EMAs (`12ema`, `26ema`) are also kept.
* **Bollinger Bands:** Upper (`upper_band`) and lower (`lower_band`) bands based on a 20-day MA and 2 standard deviations (`20sd`).
* **Exponential Moving Average (EMA):** `ema` with a center of mass `com=0.5`.
* **Momentum:** Simple momentum calculation `(close/100)-1`.

## Dependencies

You need the following Python libraries installed:
* `pandas`
* `numpy`
* `torch` (PyTorch)
* `matplotlib`
* `scikit-learn`

You can install them using pip:
```bash
pip install pandas numpy torch matplotlib scikit-learn