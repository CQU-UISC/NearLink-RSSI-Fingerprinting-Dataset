#!/usr/bin/env python
# coding: utf-8

# H-IPS implementation for mixed technology datasets
# Author: Leonardo Sestrem de Oliveira
# Revised by Song Xie on 08/08/2025
# Modified for WiFi+BLE, WiFi+SLE, BLE+SLE, WiFi+SLE+BLE datasets on 09/05/2025

import numpy as np
import pandas as pd
import scipy.optimize
from AOI import limitador
import time
import os

# Pathloss parameters for each technology
TECH_PARAMS = {
    'wifi': {'RSSI0dbm': -47.97, 'n': 1.742},
    'ble': {'RSSI0dbm': -44.19, 'n': 1.503},
    'sle': {'RSSI0dbm': -43.58, 'n': 1.563}
}

# Function to estimate distance from RSSI for a specific technology
def find_distance(TNRSSIdbm, tech, num_anchors=4):
    RSSI0dbm = TECH_PARAMS[tech]['RSSI0dbm']
    n = TECH_PARAMS[tech]['n']
    dist_buffer = []
    for i in range(num_anchors):
        dist = 10 ** ((TNRSSIdbm[i] - RSSI0dbm) / (-10 * n))
        dist_buffer.append(dist)
    return np.asarray(dist_buffer)

# Calculates Euclidean distance between each AP and estimated position
def dist_fun(APsLoc, pos):
    res = []
    for i in range(len(APsLoc)):
        res.append((APsLoc[i, :] - pos) ** 2)
    res = np.sqrt(np.sum(np.asarray(res), axis=1))
    return res

# Multilateration function for mixed technologies
def multilateracao(APsLoc, TNRSSIdbm, tech_columns):
    # tech_columns: dict mapping tech to RSSI column indices, e.g., {'wifi': [0,1,2,3], 'ble': [4,5,6,7]}
    dist_final = []
    ap_indices = []
    for tech, indices in tech_columns.items():
        dist = find_distance(TNRSSIdbm[indices], tech, num_anchors=len(indices))
        dist_final.extend(dist)
        ap_indices.extend([i for i in range(len(APsLoc))])  # Map to same APs for each tech
    dist_final = np.asarray(dist_final)
    ap_indices = np.asarray(ap_indices)
    APsLoc = APsLoc.to_numpy()

    # Cost function: minimize difference between estimated and measured distances
    cost_fun = lambda pos: np.sum((dist_fun(APsLoc[ap_indices], pos) - dist_final) ** 2)
    cond_init = np.array([2.5, 2.5])
    location = scipy.optimize.minimize(
        cost_fun,
        cond_init,
        method='l-bfgs-b',
        options={'ftol': 1e-5, 'maxiter': 1e+7}
    )
    return np.asarray([location.x])

# Scenario dimensions
largura = 5.1
comprimento = 5.1

# APs location (same for all technologies)
APsLoc = pd.DataFrame([[0, 5], [5, 5], [0, 0], [5, 0]], 
                      index=('APS1', 'APS2', 'APS3', 'APS4'), 
                      columns=['x', 'y'])

# Process each mixed dataset
datasets = {
    'wifi_ble': {'file': 'wifi_ble_dataset.csv', 'tech_columns': {'wifi': [0,1,2,3], 'ble': [4,5,6,7]}},
    'wifi_sle': {'file': 'wifi_sle_dataset.csv', 'tech_columns': {'wifi': [0,1,2,3], 'sle': [4,5,6,7]}},
    'ble_sle': {'file': 'ble_sle_dataset.csv', 'tech_columns': {'ble': [0,1,2,3], 'sle': [4,5,6,7]}},
    'wifi_sle_ble': {'file': 'wifi_sle_ble_dataset.csv', 'tech_columns': {'wifi': [0,1,2,3], 'sle': [4,5,6,7], 'ble': [8,9,10,11]}}
}

for dataset_name, config in datasets.items():
    file_path = config['file']
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，跳过")
        continue
    
    # Load RSSI values
    rssi = pd.read_csv(file_path, delimiter=',', index_col=None)
    rssi_columns = [col for col in rssi.columns if col.startswith(('wifi_rssi', 'ble_rssi', 'sle_rssi'))]
    TNRSSIdBm = rssi[rssi_columns].to_numpy()
    
    # Measure runtime
    start_time = time.time()
    
    # Estimate positions using MLT
    pos_MLT_buffer = []
    for i in range(len(TNRSSIdBm)):
        pos_MLT = multilateracao(APsLoc, TNRSSIdBm[i, :], config['tech_columns'])
        pos_MLT_buffer.append(pos_MLT)
    
    # Transform and limit estimated positions
    pos_MLT_buffer = np.asarray(pos_MLT_buffer).reshape((len(pos_MLT_buffer), 2))
    pos_MLT_buffer = limitador(pos_MLT_buffer, largura, comprimento, len(pos_MLT_buffer))
    
    # Calculate runtime
    end_time = time.time()
    runtime = end_time - start_time
    
    # True positions
    pos_geral = rssi[['x', 'y']].to_numpy()
    
    # Calculate error metrics
    dist_mlt = np.sqrt(np.sum((pos_MLT_buffer - pos_geral) ** 2, axis=1))
    mse_mlt = np.mean(dist_mlt ** 2)
    rmse_mlt = np.sqrt(mse_mlt)
    mean_dist_mlt = np.mean(dist_mlt)
    variance_mlt = np.mean((dist_mlt - mean_dist_mlt) ** 2)
    std_mlt = np.sqrt(variance_mlt)
    
    # Print error metrics and runtime
    print(f"\n{dataset_name.upper()} Error Metrics:")
    print(f"Average Euclidean Distance Error (meters): {mean_dist_mlt:.4f}")
    print(f"MSE (meters²): {mse_mlt:.4f}")
    print(f"RMSE (meters): {rmse_mlt:.4f}")
    print(f"Variance (meters²): {variance_mlt:.4f}")
    print(f"STD (meters): {std_mlt:.4f}")
    print(f"Algorithm Runtime (seconds): {runtime:.4f}")
    
    # Save estimated positions
    pos_df = pd.DataFrame({
        'x_true': pos_geral[:, 0], 'y_true': pos_geral[:, 1],
        'x_est': pos_MLT_buffer[:, 0], 'y_est': pos_MLT_buffer[:, 1]
    })
    pos_df.to_csv(f'{dataset_name}_estimated_positions.csv', index=False)
    
    # Calculate and save ECDF
    norm_MLT = np.sort(dist_mlt)
    ecdf_MLT = np.arange(1, len(dist_mlt) + 1) / len(dist_mlt)
    ecdf_data = pd.DataFrame({'Error_m': norm_MLT, 'ECDF': ecdf_MLT})
    ecdf_data.to_csv(f'ecdf_mlt_{dataset_name}.csv', index=False)