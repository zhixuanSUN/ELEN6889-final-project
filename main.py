import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from gpt_embedding_api import GPTEmbeddingAPI
from sentiment_analysis import SentimentAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load and preprocess data
def preprocess_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.lower()
    return text


data = pd.read_excel('twitter_data.xlsx')
data['cleaned_text'] = data['text'].apply(preprocess_tweet)

# Filter tweets for Google and Amazon
keywords = ['google', 'amazon', 'goog', 'amzn']
data = data[data['cleaned_text'].str.contains('|'.join(keywords))]

# Get embeddings
gpt_api = GPTEmbeddingAPI()
embeddings = gpt_api.get_embeddings(data['cleaned_text'].tolist())

# Perform sentiment analysis
sentiment_analyzer = SentimentAnalyzer()
data['sentiment_score'] = data['cleaned_text'].apply(sentiment_analyzer.get_sentiment_score)

# Summarize and analyze sentiment scores
data['date'] = pd.to_datetime(data['created_at']).dt.date
summary = data.groupby('date')['sentiment_score'].mean().reset_index()

# Load and preprocess stock price data
stock_data = pd.read_excel('stock_prices.xlsx')
stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime']).dt.date

# Merge sentiment scores with stock price data
merged_data = pd.merge(stock_data, summary, left_on='Datetime', right_on='date', how='left')
merged_data.fillna(0, inplace=True)

# Prepare data for the LSTM model
target_col = 'Close'  # or 'Open', 'High', 'Low', depending on the target
features = merged_data[['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_score']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# Train-test split
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]


# Create sequences of data for the LSTM model
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, features.columns.get_loc(target_col)])
    return np.array(X), np.array(y)


seq_len = 60
X_train, y_train = create_sequences(train_data, seq_len)
X_test, y_test = create_sequences(test_data, seq_len)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Make predictions and evaluate the model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.c_[np.zeros((predictions.shape[0], features.shape[1] - 1)), predictions])


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Calculate performance metrics
true_prices = test_data[seq_len:, features.columns.get_loc(target_col)]
predicted_prices = predictions[:, -1]

rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))
mae = mean_absolute_error(true_prices, predicted_prices)
mape = mean_absolute_percentage_error(true_prices, predicted_prices)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}%")

# Plot actual and predicted stock prices
plt.figure(figsize=(14, 6))
plt.plot(true_prices, label='Actual Prices', c='blue')
plt.plot(predicted_prices, label='Predicted Prices', c='red', linestyle='--')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()
