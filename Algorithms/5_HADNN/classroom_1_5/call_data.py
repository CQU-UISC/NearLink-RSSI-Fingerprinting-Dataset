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
        Custom dataset class for indoor localization with coordinates and RSSI values only.
        """
        folder_dir = os.path.join(path_dir, "datasets")
        folder_dir = os.path.join(folder_dir, "CustomIndoor")
        train_dir = os.path.join(folder_dir, "wifi_train_dataset_1_5m.csv")
        test_dir = os.path.join(folder_dir, "wifi_test_dataset_1_5m.csv")
        
        
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
        
        self.n_rss = 4  # Number of RSSI columns (RSSI1, RSSI2, RSSI3, RSSI4)
        
    def Divide_features(self):
        # Extract RSSI features and coordinates
        train_X = np.array(self.train_features[['rssi1', 'rssi2', 'rssi3', 'rssi4']])
        train_C = np.array(self.train_features[['x', 'y']])
        
        self.train_x = train_X
        self.train_c = train_C
        self.train_c[:, 0] = (self.train_c[:, 0] - self.x_mean) / self.x_std
        self.train_c[:, 1] = (self.train_c[:, 1] - self.y_mean) / self.y_std

        test_X = np.array(self.test_features[['rssi1', 'rssi2', 'rssi3', 'rssi4']])
        test_C = np.array(self.test_features[['x', 'y']])
        
        self.test_x = test_X
        self.test_c = test_C
        self.test_c[:, 0] = (self.test_c[:, 0] - self.x_mean) / self.x_std
        self.test_c[:, 1] = (self.test_c[:, 1] - self.y_mean) / self.y_std
        
    def Normalize_data(self):
        # Normalize RSSI values
        self.train_x = scale(self.train_x, axis=1)
        self.test_x = scale(self.test_x, axis=1)