# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon May  4 12:12:38 2020

# @author: jaehooncha

# @email: chajaehoon79@gmail.com
# """
# import numpy as np
# import pandas as pd
# import os
# from sklearn.preprocessing import scale


# class CustomIndoor(object):
#     def __init__(self, path_dir):
#         """
#         Custom dataset class for indoor localization with coordinates and RSSI values only.
#         """
#         folder_dir = os.path.join(path_dir, "datasets")
#         folder_dir = os.path.join(folder_dir, "CustomIndoor")
#         train_dir = os.path.join(folder_dir, "sle_train_dataset_2m.csv")
#         test_dir = os.path.join(folder_dir, "sle_test_dataset_2m.csv")
        
        
#         self.train_features = pd.read_csv(train_dir)
#         self.test_features = pd.read_csv(test_dir)
        
#         # Calculate mean and std for coordinate normalization
#         self.x_mean = self.train_features['x'].mean()
#         self.x_std = self.train_features['x'].std()
#         self.y_mean = self.train_features['y'].mean()
#         self.y_std = self.train_features['y'].std()
        
#         self.Divide_features()
#         self.Normalize_data()
        
#         self.num_examples = self.train_features.shape[0]
#         self.num_test_examples = self.test_features.shape[0]
        
#         self.epochs_completed = 0
#         self.index_in_epoch = 0
        
#         self.n_rss = 4  # Number of RSSI columns (RSSI1, RSSI2, RSSI3, RSSI4)
        
#     def Divide_features(self):
#         # Extract RSSI features and coordinates
#         train_X = np.array(self.train_features[['rssi1', 'rssi2', 'rssi3', 'rssi4']])
#         train_C = np.array(self.train_features[['x', 'y']])
        
#         self.train_x = train_X
#         self.train_c = train_C
#         self.train_c[:, 0] = (self.train_c[:, 0] - self.x_mean) / self.x_std
#         self.train_c[:, 1] = (self.train_c[:, 1] - self.y_mean) / self.y_std

#         test_X = np.array(self.test_features[['rssi1', 'rssi2', 'rssi3', 'rssi4']])
#         test_C = np.array(self.test_features[['x', 'y']])
        
#         self.test_x = test_X
#         self.test_c = test_C
#         self.test_c[:, 0] = (self.test_c[:, 0] - self.x_mean) / self.x_std
#         self.test_c[:, 1] = (self.test_c[:, 1] - self.y_mean) / self.y_std
        
#     def Normalize_data(self):
#         # Normalize RSSI values
#         self.train_x = scale(self.train_x, axis=1)
#         self.test_x = scale(self.test_x, axis=1)

























# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon May  4 12:12:38 2020

# @author: jaehooncha

# @email: chajaehoon79@gmail.com
# """
# import numpy as np
# import pandas as pd
# import os
# from sklearn.preprocessing import scale


# class CustomIndoor(object):
#     def __init__(self, path_dir):
#         """
#         Custom dataset class for indoor localization with coordinates and RSSI values only.
#         """
#         folder_dir = os.path.join(path_dir, "datasets")
#         folder_dir = os.path.join(folder_dir, "CustomIndoor")

#         # train_dir = os.path.join(folder_dir, "train_wifi_ble_dataset_2m.csv")
#         # test_dir = os.path.join(folder_dir, "test_wifi_ble_dataset_2m.csv")
        
#         # train_dir = os.path.join(folder_dir, "train_wifi_sle_dataset_2m.csv")
#         # test_dir = os.path.join(folder_dir, "test_wifi_sle_dataset_2m.csv")

#         train_dir = os.path.join(folder_dir, "train_ble_sle_dataset_2m.csv")
#         test_dir = os.path.join(folder_dir, "test_ble_sle_dataset_2m.csv")






#         self.train_features = pd.read_csv(train_dir)
#         self.test_features = pd.read_csv(test_dir)
        
#         # Calculate mean and std for coordinate normalization
#         self.x_mean = self.train_features['x'].mean()
#         self.x_std = self.train_features['x'].std()
#         self.y_mean = self.train_features['y'].mean()
#         self.y_std = self.train_features['y'].std()
        
#         self.Divide_features()
#         self.Normalize_data()
        
#         self.num_examples = self.train_features.shape[0]
#         self.num_test_examples = self.test_features.shape[0]
        
#         self.epochs_completed = 0
#         self.index_in_epoch = 0
        
#         self.n_rss = 8  # Number of RSSI columns (wifi_rssi1-4, ble_rssi1-4)
        
#     def Divide_features(self):
#         # Extract RSSI features and coordinates

#         # train_X = np.array(self.train_features[['wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4']])
#         # train_X = np.array(self.train_features[['wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4']])
#         train_X = np.array(self.train_features[['ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4', 'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4']])

#         train_C = np.array(self.train_features[['x', 'y']])
        
#         self.train_x = train_X
#         self.train_c = train_C
#         self.train_c[:, 0] = (self.train_c[:, 0] - self.x_mean) / self.x_std
#         self.train_c[:, 1] = (self.train_c[:, 1] - self.y_mean) / self.y_std

#         # test_X = np.array(self.test_features[['wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4']])
#         # test_X = np.array(self.test_features[['wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4']])
#         test_X = np.array(self.test_features[['ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4', 'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4']])

#         test_C = np.array(self.test_features[['x', 'y']])
        
#         self.test_x = test_X
#         self.test_c = test_C
#         self.test_c[:, 0] = (self.test_c[:, 0] - self.x_mean) / self.x_std
#         self.test_c[:, 1] = (self.test_c[:, 1] - self.y_mean) / self.y_std
        
#     def Normalize_data(self):
#         # Normalize RSSI values
#         self.train_x = scale(self.train_x, axis=1)
#         self.test_x = scale(self.test_x, axis=1)
















# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon May  4 12:12:38 2020

# @author: jaehooncha

# @email: chajaehoon79@gmail.com
# """
# import numpy as np
# import pandas as pd
# import os
# from sklearn.preprocessing import scale


# class CustomIndoor(object):
#     def __init__(self, path_dir):
#         """
#         Custom dataset class for indoor localization with coordinates and WiFi, SLE, BLE RSSI values.
#         """
#         folder_dir = os.path.join(path_dir, "datasets")
#         folder_dir = os.path.join(folder_dir, "CustomIndoor")
#         train_dir = os.path.join(folder_dir, "train_wifi_sle_ble_dataset_2m.csv")
#         test_dir = os.path.join(folder_dir, "test_wifi_sle_ble_dataset_2m.csv")
        
#         self.train_features = pd.read_csv(train_dir)
#         self.test_features = pd.read_csv(test_dir)
        
#         # Calculate mean and std for coordinate normalization
#         self.x_mean = self.train_features['x'].mean()
#         self.x_std = self.train_features['x'].std()
#         self.y_mean = self.train_features['y'].mean()
#         self.y_std = self.train_features['y'].std()
        
#         self.Divide_features()
#         self.Normalize_data()
        
#         self.num_examples = self.train_features.shape[0]
#         self.num_test_examples = self.test_features.shape[0]
        
#         self.epochs_completed = 0
#         self.index_in_epoch = 0
        
#         self.n_rss = 12  # Number of RSSI columns (WiFi: 4, SLE: 4, BLE: 4)
        
#     def Divide_features(self):
#         # Extract RSSI features (WiFi, SLE, BLE) and coordinates
#         feature_columns = [
#             'wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4',
#             'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4',
#             'ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4'
#         ]
#         train_X = np.array(self.train_features[feature_columns])
#         train_C = np.array(self.train_features[['x', 'y']])
        
#         self.train_x = train_X
#         self.train_c = train_C
#         self.train_c[:, 0] = (self.train_c[:, 0] - self.x_mean) / self.x_std
#         self.train_c[:, 1] = (self.train_c[:, 1] - self.y_mean) / self.y_std

#         test_X = np.array(self.test_features[feature_columns])
#         test_C = np.array(self.test_features[['x', 'y']])
        
#         self.test_x = test_X
#         self.test_c = test_C
#         self.test_c[:, 0] = (self.test_c[:, 0] - self.x_mean) / self.x_std
#         self.test_c[:, 1] = (self.test_c[:, 1] - self.y_mean) / self.y_std
        
#     def Normalize_data(self):
#         # Normalize RSSI values
#         self.train_x = scale(self.train_x, axis=1)
#         self.test_x = scale(self.test_x, axis=1)




















# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon May  4 12:12:38 2020

# @author: jaehooncha

# @email: chajaehoon79@gmail.com
# """
# import numpy as np
# import pandas as pd
# import os
# from sklearn.preprocessing import scale


# class CustomIndoor(object):
#     def __init__(self, path_dir):
#         """
#         Custom dataset class for indoor localization with coordinates and RSSI values only.
#         """
#         folder_dir = os.path.join(path_dir, "datasets")
#         folder_dir = os.path.join(folder_dir, "CustomIndoor")
#         train_dir = os.path.join(folder_dir, "sle_train_dataset_heli.csv")
#         test_dir = os.path.join(folder_dir, "sle_test_dataset_heli.csv")
        
        
#         self.train_features = pd.read_csv(train_dir)
#         self.test_features = pd.read_csv(test_dir)
        
#         # Calculate mean and std for coordinate normalization
#         self.x_mean = self.train_features['x'].mean()
#         self.x_std = self.train_features['x'].std()
#         self.y_mean = self.train_features['y'].mean()
#         self.y_std = self.train_features['y'].std()
        
#         self.Divide_features()
#         self.Normalize_data()
        
#         self.num_examples = self.train_features.shape[0]
#         self.num_test_examples = self.test_features.shape[0]
        
#         self.epochs_completed = 0
#         self.index_in_epoch = 0
        
#         self.n_rss = 8  # Number of RSSI columns (RSSI1, RSSI2, RSSI3, RSSI4)
        
#     def Divide_features(self):
#         # Extract RSSI features and coordinates
#         train_X = np.array(self.train_features[['rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8']])
#         train_C = np.array(self.train_features[['x', 'y']])
        
#         self.train_x = train_X
#         self.train_c = train_C
#         self.train_c[:, 0] = (self.train_c[:, 0] - self.x_mean) / self.x_std
#         self.train_c[:, 1] = (self.train_c[:, 1] - self.y_mean) / self.y_std

#         test_X = np.array(self.test_features[['rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8']])
#         test_C = np.array(self.test_features[['x', 'y']])
        
#         self.test_x = test_X
#         self.test_c = test_C
#         self.test_c[:, 0] = (self.test_c[:, 0] - self.x_mean) / self.x_std
#         self.test_c[:, 1] = (self.test_c[:, 1] - self.y_mean) / self.y_std
        
#     def Normalize_data(self):
#         # Normalize RSSI values
#         self.train_x = scale(self.train_x, axis=1)
#         self.test_x = scale(self.test_x, axis=1)




























# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon May  4 12:12:38 2020

# @author: jaehooncha

# @email: chajaehoon79@gmail.com
# """
# import numpy as np
# import pandas as pd
# import os
# from sklearn.preprocessing import scale


# class CustomIndoor(object):
#     def __init__(self, path_dir):
#         """
#         Custom dataset class for indoor localization with coordinates and RSSI values only.
#         """
#         folder_dir = os.path.join(path_dir, "datasets")
#         folder_dir = os.path.join(folder_dir, "CustomIndoor")

#         train_dir = os.path.join(folder_dir, "train_wifi_ble_dataset_heli.csv")
#         test_dir = os.path.join(folder_dir, "test_wifi_ble_dataset_heli.csv")
        
#         # train_dir = os.path.join(folder_dir, "train_wifi_sle_dataset_heli.csv")
#         # test_dir = os.path.join(folder_dir, "test_wifi_sle_dataset_heli.csv")

#         # train_dir = os.path.join(folder_dir, "train_ble_sle_dataset_heli.csv")
#         # test_dir = os.path.join(folder_dir, "test_ble_sle_dataset_heli.csv")






#         self.train_features = pd.read_csv(train_dir)
#         self.test_features = pd.read_csv(test_dir)
        
#         # Calculate mean and std for coordinate normalization
#         self.x_mean = self.train_features['x'].mean()
#         self.x_std = self.train_features['x'].std()
#         self.y_mean = self.train_features['y'].mean()
#         self.y_std = self.train_features['y'].std()
        
#         self.Divide_features()
#         self.Normalize_data()
        
#         self.num_examples = self.train_features.shape[0]
#         self.num_test_examples = self.test_features.shape[0]
        
#         self.epochs_completed = 0
#         self.index_in_epoch = 0
        
#         self.n_rss = 16  # Number of RSSI columns (wifi_rssi1-4, ble_rssi1-4)
        
#     def Divide_features(self):
#         # Extract RSSI features and coordinates

#         train_X = np.array(self.train_features[['wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'wifi_rssi5', 'wifi_rssi6', 'wifi_rssi7', 'wifi_rssi8',
#                                                 'ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4', 'ble_rssi5', 'ble_rssi6', 'ble_rssi7', 'ble_rssi8']])
#         # train_X = np.array(self.train_features[['wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'wifi_rssi5', 'wifi_rssi6', 'wifi_rssi7', 'wifi_rssi8',
#         #                                         'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4', 'sle_rssi5', 'sle_rssi6', 'sle_rssi7', 'sle_rssi8']])
#         # train_X = np.array(self.train_features[['ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4', 'ble_rssi5', 'ble_rssi6', 'ble_rssi7', 'ble_rssi8', 
#         #                                         'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4', 'sle_rssi5', 'sle_rssi6', 'sle_rssi7', 'sle_rssi8']])

#         train_C = np.array(self.train_features[['x', 'y']])
        
#         self.train_x = train_X
#         self.train_c = train_C
#         self.train_c[:, 0] = (self.train_c[:, 0] - self.x_mean) / self.x_std
#         self.train_c[:, 1] = (self.train_c[:, 1] - self.y_mean) / self.y_std

#         test_X = np.array(self.test_features[['wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'wifi_rssi5', 'wifi_rssi6', 'wifi_rssi7', 'wifi_rssi8',
#                                               'ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4', 'ble_rssi5', 'ble_rssi6', 'ble_rssi7', 'ble_rssi8']])
#         # test_X = np.array(self.test_features[['wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'wifi_rssi5', 'wifi_rssi6', 'wifi_rssi7', 'wifi_rssi8', 
#         #                                       'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4', 'sle_rssi5', 'sle_rssi6', 'sle_rssi7', 'sle_rssi8']])
#         # test_X = np.array(self.test_features[['ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4', 'ble_rssi5', 'ble_rssi6', 'ble_rssi7', 'ble_rssi8',
#         #                                       'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4', 'sle_rssi5', 'sle_rssi6', 'sle_rssi7', 'sle_rssi8']])

#         test_C = np.array(self.test_features[['x', 'y']])
        
#         self.test_x = test_X
#         self.test_c = test_C
#         self.test_c[:, 0] = (self.test_c[:, 0] - self.x_mean) / self.x_std
#         self.test_c[:, 1] = (self.test_c[:, 1] - self.y_mean) / self.y_std
        
#     def Normalize_data(self):
#         # Normalize RSSI values
#         self.train_x = scale(self.train_x, axis=1)
#         self.test_x = scale(self.test_x, axis=1)





















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:12:38 2020

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import scale


class CustomIndoor(object):
    def __init__(self, path_dir):
        """
        Custom dataset class for indoor localization with coordinates and WiFi, SLE, BLE RSSI values.
        """
        folder_dir = os.path.join(path_dir, "datasets")
        folder_dir = os.path.join(folder_dir, "CustomIndoor")
        train_dir = os.path.join(folder_dir, "train_wifi_sle_ble_dataset_heli.csv")
        test_dir = os.path.join(folder_dir, "test_wifi_sle_ble_dataset_heli.csv")
        
        self.train_features = pd.read_csv(train_dir)
        self.test_features = pd.read_csv(test_dir)
        
        # Calculate mean and std for coordinate normalization
        self.x_mean = self.train_features['x'].mean()
        self.x_std = self.train_features['x'].std()
        self.y_mean = self.train_features['y'].mean()
        self.y_std = self.train_features['y'].std()
        
        self.Divide_features()
        self.Normalize_data()
        
        self.num_examples = self.train_features.shape[0]
        self.num_test_examples = self.test_features.shape[0]
        
        self.epochs_completed = 0
        self.index_in_epoch = 0
        
        self.n_rss = 24  # Number of RSSI columns (WiFi: 4, SLE: 4, BLE: 4)
        
    def Divide_features(self):
        # Extract RSSI features (WiFi, SLE, BLE) and coordinates
        feature_columns = [
            'wifi_rssi1', 'wifi_rssi2', 'wifi_rssi3', 'wifi_rssi4', 'wifi_rssi5', 'wifi_rssi6', 'wifi_rssi7', 'wifi_rssi8',
            'sle_rssi1', 'sle_rssi2', 'sle_rssi3', 'sle_rssi4', 'sle_rssi5', 'sle_rssi6', 'sle_rssi7', 'sle_rssi8',
            'ble_rssi1', 'ble_rssi2', 'ble_rssi3', 'ble_rssi4', 'ble_rssi5', 'ble_rssi6', 'ble_rssi7', 'ble_rssi8'
        ]
        train_X = np.array(self.train_features[feature_columns])
        train_C = np.array(self.train_features[['x', 'y']])
        
        self.train_x = train_X
        self.train_c = train_C
        self.train_c[:, 0] = (self.train_c[:, 0] - self.x_mean) / self.x_std
        self.train_c[:, 1] = (self.train_c[:, 1] - self.y_mean) / self.y_std

        test_X = np.array(self.test_features[feature_columns])
        test_C = np.array(self.test_features[['x', 'y']])
        
        self.test_x = test_X
        self.test_c = test_C
        self.test_c[:, 0] = (self.test_c[:, 0] - self.x_mean) / self.x_std
        self.test_c[:, 1] = (self.test_c[:, 1] - self.y_mean) / self.y_std
        
    def Normalize_data(self):
        # Normalize RSSI values
        self.train_x = scale(self.train_x, axis=1)
        self.test_x = scale(self.test_x, axis=1)
