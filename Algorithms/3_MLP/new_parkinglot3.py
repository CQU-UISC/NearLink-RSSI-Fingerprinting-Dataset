import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import time
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Force CPU usage (disable GPU if available)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load data from CSV files in the current folder
df_train = pd.read_csv('train_wifi_sle_ble_dataset_1_5m.csv')
df_test = pd.read_csv('test_wifi_sle_ble_dataset_1_5m.csv')

# Prepare features (RSSI) and targets (x, y)
X_train = df_train[['wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'wifi_rssi5', 'wifi_rssi6', 'wifi_rssi7', 'wifi_rssi8',
                     'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4', 'sle_rssi5', 'sle_rssi6', 'sle_rssi7', 'sle_rssi8',
                       'ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4', 'ble_rssi5', 'ble_rssi6', 'ble_rssi7', 'ble_rssi8']].values
y_train = df_train[['x', 'y']].values
X_test = df_test[['wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'wifi_rssi5', 'wifi_rssi6', 'wifi_rssi7', 'wifi_rssi8',
                   'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4', 'sle_rssi5', 'sle_rssi6', 'sle_rssi7', 'sle_rssi8',
                     'ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4', 'ble_rssi5', 'ble_rssi6', 'ble_rssi7', 'ble_rssi8']].values
y_test = df_test[['x', 'y']].values

# Scale the data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)  # For consistency, but we'll inverse transform predictions

# Build the MLP model based on the benchmark MLP from the paper
model = Sequential([
    Dense(256, activation='relu', input_shape=(24,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(2, activation='linear')  # Linear activation for regression output (x, y)
])

optimizer = Adam(learning_rate=5e-4)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Record start time
start_time = time.time()

# Train the model with 10% validation split and 1000 epochs (no early stopping)
history = model.fit(
    X_train_scaled, y_train_scaled,
    batch_size=64,
    epochs=1000,  # Fixed to 1000 epochs as per the paper's Fig. 2
    validation_split=0.1,  # 10% validation set as per common practice
    verbose=1
)

# Record end time and calculate runtime
end_time = time.time()
runtime = end_time - start_time
print(f"Algorithm runtime: {runtime:.2f} seconds")

# Predict on test set
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test_scaled)  # For consistency

# Calculate Euclidean distances for each test point
errors = []
for i in range(len(y_test_orig)):
    dist = np.sqrt(
        (y_test_orig[i, 0] - y_pred[i, 0]) ** 2 +
        (y_test_orig[i, 1] - y_pred[i, 1]) ** 2
    )
    errors.append(dist)

# Calculate Mean Euclidean Distance Error
mean_euclidean_error = np.mean(errors)
print(f"Average Euclidean Distance Localization Error: {mean_euclidean_error:.4f} meters")

# Calculate RMSE
rmse = np.sqrt(np.mean(np.array(errors) ** 2))
print(f"Root Mean Square Error (RMSE): {rmse:.4f} meters")

# Calculate Variance and Standard Deviation
variance = np.var(errors)
std = np.std(errors)
print(f"Variance: {variance:.4f} meters^2")
print(f"Standard Deviation (STD): {std:.4f} meters")

# Save errors to CSV file
error_df = pd.DataFrame({
    'Test_Point_Index': range(len(errors)),
    'True_X': y_test_orig[:, 0],
    'True_Y': y_test_orig[:, 1],
    'Predicted_X': y_pred[:, 0],
    'Predicted_Y': y_pred[:, 1],
    'Euclidean_Error': errors
})
error_df.to_csv('test_point_localization_errors.csv', index=False)
print("Localization errors saved to 'test_point_localization_errors.csv'")