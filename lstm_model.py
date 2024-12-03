# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset from GitHub
def load_data():
    url = "https://raw.githubusercontent.com/srinivem/Federal-Interest-Rates-Prediction-for-2025/lstm_model/finaldata.csv"
    print("Loading data...")
    data = pd.read_csv(url)
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns.")
    return data

# Preprocess the data
def preprocess_data(data):
    # Add lagged features
    data['FEDFUNDS_Lag1'] = data['FEDFUNDS'].shift(1)
    data['FEDFUNDS_Lag2'] = data['FEDFUNDS'].shift(2)
    data = data.dropna()  # Drop rows with NaN values

    # Extract month and one-hot encode
    data['Month'] = pd.to_datetime(data['Date']).dt.month
    data = pd.get_dummies(data, columns=['Month'], drop_first=True)

    # Define features and target
    TARGET = 'FEDFUNDS'
    FEATURES = ['Inflation Rate', 'Unemployment Rate', 'Bonds Yield', 
                'FEDFUNDS_Lag1', 'FEDFUNDS_Lag2'] + [col for col in data.columns if col.startswith('Month_')]

    # Scale features and target
    scaler = MinMaxScaler()
    features = scaler.fit_transform(data[FEATURES].values)
    target_scaler = MinMaxScaler()
    target = target_scaler.fit_transform(data[TARGET].values.reshape(-1, 1)).flatten()

    return features, target, scaler, target_scaler, data

# Prepare sequences for LSTM
def prepare_sequences(features, target, seq_len=12):
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])  # Last `seq_len` steps as input
        y.append(target[i])  # Current value as target
    return np.array(X), np.array(y)

# Build the LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dropout(0.4),  # Regularization to prevent overfitting
        Dense(1, kernel_regularizer=l2(0.01))  # L2 regularization
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    return model

# Train the model
def train_model(model, X_train, y_train):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])
    return model, history

# Evaluate the model
def evaluate_model(model, X_test, y_test, target_scaler):
    predictions = model.predict(X_test)
    predictions_rescaled = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_rescaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    print(f"Mean Squared Error (Rescaled): {mse:.4f}")

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled, label='Actual', color='blue')
    plt.plot(predictions_rescaled, label='Predicted', color='orange')
    plt.title("LSTM Predictions vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Federal Funds Rate")
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    # Load and preprocess data
    data = load_data()
    features, target, scaler, target_scaler, processed_data = preprocess_data(data)

    # Prepare sequences and split data
    SEQ_LEN = 12
    X, y = prepare_sequences(features, target, seq_len=SEQ_LEN)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train the model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model, history = train_model(model, X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, target_scaler)
