#!/usr/bin/env python
# coding: utf-8

# # H-IPS implementation
# ## Author : Leonardo Sestrem de Oliveira
# ## Date: 07/10/2021
# ## Indoor Positioning System combining Multilateration and Fingerprinting

# Revised by Song Xie on 08/08/2025


# Import section
import numpy as np
import pandas as pd
import scipy.optimize
from AOI import limitador
import time

# # Function to estimate distance between TN and AP through measured RSSI
# def find_distance(TNRSSIdbm):
#     RSSI0dbm = -47.97    # RSSI at reference distance         **********************Need to change***********************
#     s = 0               # Shadowing value in dB
#     n = 1.742           # Pathloss exponent                    **********************Need to change***********************
#     d0 = 1.0            # Reference distance
#     dist_buffer = []
    
#     for i in range(4):  # Adjusted for 4 RSSI columns in test_database.csv **********************Need to change***********************
#         dist = 10**(-((TNRSSIdbm[i] - RSSI0dbm + s) / (10 * n)) + np.log10(d0))
#         dist_buffer.append(dist)
        
#     return np.asarray(dist_buffer)

def find_distance(TNRSSIdbm):
    RSSI0dbm = -47.97    # Constant C (reference RSSI in dBm)
    n = 1.742     # Pathloss exponent
    dist_buffer = []
    
    for i in range(4):  # Process 4 RSSI values
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
    cond_init = np.array([2.5, 2.5])
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
largura = 5                             #**********************Need to change***********************
comprimento = 5                           #**********************Need to change***********************

# APs location (adjusted to 4 APs to match test_database.csv RSSI columns)
APsLoc = pd.DataFrame([[0, 5], [5, 5], [0, 0], [5, 0]],                  #**********************Need to change***********************
                     index=('APS1', 'APS2', 'APS3', 'APS4'),             #**********************Need to change***********************
                     columns=['x', 'y']) 

# Load RSSI values from test_database.csv
rssi = pd.read_csv("wifi_test_dataset_1_5m.csv", delimiter=',', index_col=None)   #**********************Need to change***********************
# TNRSSIdBm = rssi[['rssi1', 'rssi2', 'rssi3', 'rssi4']].to_numpy()        #**********************Need to change***********************

TNRSSIdBm = rssi[['rssi_1', 'rssi_2', 'rssi_3', 'rssi_4']].to_numpy() 
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
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(APsLoc.x, APsLoc.y, 'r*', label='AP', markersize=5)
# plt.plot(pos_geral[:, 0], pos_geral[:, 1], 'ko-', label='pos_true', markersize=5)
# plt.plot(pos_MLT_buffer[:, 0], pos_MLT_buffer[:, 1], 'mo', label='MLT', markersize=2)

# for a, b, c in zip(APsLoc.x, APsLoc.y, APsLoc.index):
#     plt.annotate(c, (a, b), textcoords="offset points", xytext=(0, 5), ha='center')
    
# plt.xticks(np.arange(0, 5.1, 0.5))                   #**********************Need to change***********************
# plt.yticks(np.arange(0, 5.1, 0.5))                   #**********************Need to change***********************
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.legend()
# plt.grid()
# plt.show()

# Calculate and plot empirical ECDF for MLT
norm_MLT = np.sort(dist_mlt)
ecdf_MLT = np.arange(1, len(dist_mlt) + 1) / len(dist_mlt)

# plt.figure(2)
# plt.step(norm_MLT, ecdf_MLT, 'r-', label='MLT', where='post')
# plt.title('Empirical ECDF')
# plt.xlabel('Error [m]')
# plt.ylabel('ECDF')
# plt.legend()
# plt.grid()
# plt.show()

# 保存每个点的定位误差到CSV
error_data = pd.DataFrame({
    'x_true': pos_geral[:, 0],
    'y_true': pos_geral[:, 1],
    'x_est': pos_MLT_buffer[:, 0],
    'y_est': pos_MLT_buffer[:, 1],
    'Error_m': dist_mlt
})
error_data.to_csv('pointwise_errors_mlt.csv', index=False)

print("每个点的定位误差已保存至 pointwise_errors_mlt.csv")

# Export ECDF values to a CSV file
ecdf_data = pd.DataFrame({'Error_m': norm_MLT, 'ECDF': ecdf_MLT})
ecdf_data.to_csv('ecdf_mlt.csv', index=False)