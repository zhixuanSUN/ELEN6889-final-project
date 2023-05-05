#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from datetime import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


training_set = pd.read_csv("apple.csv")
training_set


# In[ ]:


df = training_set[['Close', 'Adj Close', 'Volume', 'Sentiment']]
df['Close'] = df['Close'].shift(-1)
df.drop(df.tail(1).index,inplace=True)
df


# In[1]:


X = df[['Close', 'Volume', 'Sentiment']]
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

y = df['Close'].values
sc2 = MinMaxScaler(feature_range=(0,1))
y = y.reshape(-1, 1)
y = sc2.fit_transform(y)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = False, stratify = None)


# In[25]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=15)


# In[26]:


y_pred = model.predict(X_test)
y_pred = sc2.inverse_transform(y_pred)
y_true = sc2.inverse_transform(y_test)

from sklearn.metrics import mean_squared_error

mean_squared_error(y_true, y_pred, squared=False)


# In[27]:


y_pred = model.predict(X_test)
y_pred = sc2.inverse_transform(y_pred)
y_true = sc2.inverse_transform(y_test)
# Visualize the data

plt.figure(figsize=(16,6))
plt.plot(y_true)
plt.plot(y_pred)
plt.show()


# In[ ]:




