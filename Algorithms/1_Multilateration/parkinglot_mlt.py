#!/usr/bin/env python
# coding: utf-8

# H-IPS implementation
# Author: Leonardo Sestrem de Oliveira
# Date: 07/10/2021
# Indoor Positioning System combining Multilateration and Fingerprinting

# Revised by Song Xie on 08/08/2025
# Updated for 8 anchor points on 09/06/2025

# Import section
import numpy as np
import pandas as pd
import scipy.optimize
from AOI import limitador
import time

def find_distance(TNRSSIdbm):
    RSSI0dbm = -44.05    # Constant C (reference RSSI in dBm)
    n = 1.905     # Pathloss exponent
    dist_buffer = []
    
    for i in range(8):  # Process 8 RSSI values
        dist = 10 ** ((TNRSSIdbm[i] - RSSI0dbm) / (-10 * n))
        dist_buffer.append(dist)
        
    return np.asarray(dist_buffer)

# Calculates Euclidean distance between each AP and estimated position
def dist_fun(APsLoc, pos):
    res = []
    for i in range(len(APsLoc)):
        res.append(list((APsLoc[i, :] - pos) ** 2))
    res = np.sqrt(np.sum(np.asarray(res), axis=1))
    return res

# Multilateration function
def multilateracao(APsLoc, TNRSSIdbm):
    dist_final = find_distance(TNRSSIdbm)
    dist_final = np.transpose(dist_final)
    pos_est_mlt = []
    APsLoc = APsLoc.to_numpy()
    cost_fun = lambda pos: np.sum((dist_fun(APsLoc, pos) - dist_final) ** 2)
    cond_init = np.array([11.25, 14.25])  # Adjusted initial guess to center of new area
    location = scipy.optimize.minimize(
        cost_fun,
        cond_init,
        method='l-bfgs-b',
        options={'ftol': 1e-5, 'maxiter': 1e+7}
    )
    posicao = location.x
    pos_est_mlt.append(list(posicao))
    return np.asarray(pos_est_mlt)

# Scenario dimensions
largura = 22.5    # Updated to match new AP coordinates
comprimento = 28.5  # Updated to match new AP coordinates

# APs location (updated to 8 APs)
APsLoc = pd.DataFrame([
    [0, 28.5], [11.25, 28.5], [22.5, 28.5], [0, 14.25], 
    [22.5, 14.25], [0, 0], [11.25, 0], [22.5, 0]
], index=('APS1', 'APS2', 'APS3', 'APS4', 'APS5', 'APS6', 'APS7', 'APS8'), 
columns=['x', 'y'])

# Load RSSI values from test_database.csv
rssi = pd.read_csv("sle_test_dataset_1_5m.csv", delimiter=',', index_col=None)
TNRSSIdBm = rssi[['rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8']].to_numpy()

# Measure runtime for MLT algorithm
start_time = time.time()

# Estimate positions using MLT
pos_MLT_buffer = []
for i in range(len(TNRSSIdBm)):
    pos_MLT = multilateracao(APsLoc, TNRSSIdBm[i, :])
    pos_MLT_buffer.append(pos_MLT)

# Transform and limit estimated positions
pos_MLT_buffer = np.asarray(pos_MLT_buffer).reshape((len(pos_MLT_buffer), 2))
pos_MLT_buffer = limitador(pos_MLT_buffer, largura, comprimento, len(pos_MLT_buffer))

# Calculate runtime
end_time = time.time()
runtime = end_time - start_time

# True positions from test_database.csv
pos_geral = rssi[['x', 'y']].to_numpy()

# Calculate mean square error for MLT
dist_mlt = []
for i in range(len(pos_MLT_buffer)):
    dist_mlt.append(list((pos_MLT_buffer[i, :] - pos_geral[i]) ** 2))
dist_mlt = np.sqrt(np.sum(np.asarray(dist_mlt), axis=1))

# Calculate MSE, RMSE, Variance, and STD
mse_mlt = np.mean(dist_mlt ** 2)
rmse_mlt = np.sqrt(mse_mlt)
mean_dist_mlt = np.mean(dist_mlt)
variance_mlt = np.mean((dist_mlt - mean_dist_mlt) ** 2)
std_mlt = np.sqrt(variance_mlt)

# Print error metrics and runtime
print("MLT Error Metrics:")
print(f"Average Euclidean Distance Error (meters): {mean_dist_mlt:.4f}")
print(f"MSE (meters²): {mse_mlt:.4f}")
print(f"RMSE (meters): {rmse_mlt:.4f}")
print(f"Variance (meters²): {variance_mlt:.4f}")
print(f"STD (meters): {std_mlt:.4f}")
print(f"Algorithm Runtime (seconds): {runtime:.4f}")

# Plot the test trajectory
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(APsLoc.x, APsLoc.y, 'r*', label='AP', markersize=5)
plt.plot(pos_geral[:, 0], pos_geral[:, 1], 'ko-', label='pos_true', markersize=5)
plt.plot(pos_MLT_buffer[:, 0], pos_MLT_buffer[:, 1], 'mo', label='MLT', markersize=2)

for a, b, c in zip(APsLoc.x, APsLoc.y, APsLoc.index):
    plt.annotate(c, (a, b), textcoords="offset points", xytext=(0, 5), ha='center')
    
plt.xticks(np.arange(0, 22.6, 1.5))  # Adjusted for new width
plt.yticks(np.arange(0, 28.6, 1.5))  # Adjusted for new length
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.grid()
plt.show()

# Calculate and plot empirical ECDF for MLT
norm_MLT = np.sort(dist_mlt)
ecdf_MLT = np.arange(1, len(dist_mlt) + 1) / len(dist_mlt)

plt.figure(2)
plt.step(norm_MLT, ecdf_MLT, 'r-', label='MLT', where='post')
plt.title('Empirical ECDF')
plt.xlabel('Error [m]')
plt.ylabel('ECDF')
plt.legend()
plt.grid()
plt.show()

# Export ECDF values to a CSV file
ecdf_data = pd.DataFrame({'Error_m': norm_MLT, 'ECDF': ecdf_MLT})
ecdf_data.to_csv('ecdf_mlt.csv', index=False)