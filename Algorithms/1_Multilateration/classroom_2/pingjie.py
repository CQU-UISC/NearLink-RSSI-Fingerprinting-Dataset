import pandas as pd
import os

# 定义文件路径（当前文件夹）
ble_file = 'ble_test_dataset_2m.csv'
sle_file = 'sle_test_dataset_2m.csv'
wifi_file = 'wifi_test_dataset_2m.csv'

# 检查文件是否存在
for file in [ble_file, sle_file, wifi_file]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"文件 {file} 不存在于当前文件夹")

# 读取数据并重命名RSSI列以区分技术
df_ble = pd.read_csv(ble_file).rename(columns={
    'rssi1': 'ble_rssi1', 'rssi2': 'ble_rssi2', 
    'rssi3': 'ble_rssi3', 'rssi4': 'ble_rssi4'
})
df_sle = pd.read_csv(sle_file).rename(columns={
    'rssi1': 'sle_rssi1', 'rssi2': 'sle_rssi2', 
    'rssi3': 'sle_rssi3', 'rssi4': 'sle_rssi4'
})
df_wifi = pd.read_csv(wifi_file).rename(columns={
    'rssi1': 'wifi_rssi1', 'rssi2': 'wifi_rssi2', 
    'rssi3': 'wifi_rssi3', 'rssi4': 'wifi_rssi4'
})

# 1. WiFi + BLE
df_wifi_ble = pd.merge(df_wifi, df_ble, on=['x', 'y'])
df_wifi_ble.to_csv('wifi_ble_dataset.csv', index=False)
print("已生成 wifi_ble_dataset.csv")

# 2. WiFi + SLE
df_wifi_sle = pd.merge(df_wifi, df_sle, on=['x', 'y'])
df_wifi_sle.to_csv('wifi_sle_dataset.csv', index=False)
print("已生成 wifi_sle_dataset.csv")

# 3. BLE + SLE
df_ble_sle = pd.merge(df_ble, df_sle, on=['x', 'y'])
df_ble_sle.to_csv('ble_sle_dataset.csv', index=False)
print("已生成 ble_sle_dataset.csv")

# 4. WiFi + SLE + BLE
df_all = pd.merge(pd.merge(df_wifi, df_sle, on=['x', 'y']), df_ble, on=['x', 'y'])
df_all.to_csv('wifi_sle_ble_dataset.csv', index=False)
print("已生成 wifi_sle_ble_dataset.csv")