# Stock Price Prediction using Linear Regression

# Install yfinance
!pip install yfinance

# Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Download stock data
stock_data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Display first 5 rows
print(stock_data.head())

# Plot closing price
plt.figure(figsize=(10,5))
plt.plot(stock_data['Close'])
plt.title("Apple Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Create prediction column
stock_data['Prediction'] = stock_data[['Close']].shift(-30)

# Prepare data
X = np.array(stock_data[['Close']])[:-30]
y = np.array(stock_data['Prediction'])[:-30]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict future prices
future_prices = np.array(stock_data[['Close']])[-30:]

predictions = model.predict(future_prices)

# Print predictions
print("\nPredicted Prices:")
print(predictions)

# Plot predictions
plt.figure(figsize=(10,5))
plt.plot(predictions)
plt.title("Predicted Future Stock Prices")
plt.xlabel("Days")
plt.ylabel("Predicted Price")
plt.show()