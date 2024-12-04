import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import pickle

# Load your dataset
def load_data():
    file_path = 'finaldata.csv'  # Update to your actual path
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('MS').dropna()  # Ensure monthly frequency and no missing values
    return df

fedfunds = load_data()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(fedfunds.values)

# Prepare sequences for LSTM
seq_len = 12
X, y = [], []
for i in range(seq_len, len(scaled_data)):
    X.append(scaled_data[i - seq_len:i])
    y.append(scaled_data[i, 0])  # Assuming 'FEDFUNDS' is the target column

X, y = np.array(X), np.array(y)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build and train the LSTM model
model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(seq_len, X.shape[2])),
    Dropout(0.4),
    Dense(1, kernel_regularizer=l2(0.01))
])
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Save the model and scaler
model.save('lstm_model.h5')  # Save model
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
