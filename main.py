# Import modules
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Disable warns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Helper function
# Construct np array from dataframe
def make_dataset(df):
    x, y = [], []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])

    x, y = np.array(x), np.array(y)
    return x, y

# Read and form train/test frames
df = pd.read_csv("./data_and_tests/ABR.csv")
df = df["Open"].values.reshape(-1, 1)
dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8):])

# Scale/reshape frames
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

# Create dataset arrays
x_train, y_train = make_dataset(dataset_train)
x_test, y_test = make_dataset(dataset_test)

# Define RNN model (comment out on testing)
model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile, train, and save (comment out on testing)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(x_train, y_train, epochs=50, batch_size=32)
model.save("stock_prediction.keras")

# Load model and sample CSV
model = load_model("stock_prediction.keras")
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
fig, ax = plt.subplots(figsize=(16,8))
fig.canvas.manager.set_window_title("StockPredictor")
ax.set_facecolor("#2d313b")
ax.set_title("Stock Prediction")
ax.plot(y_test_scaled, color="magenta", label="Original price")
plt.plot(pred, color="lime", label="Price prediction")
plt.legend()
plt.show()
