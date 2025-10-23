import tensorflow as tf
# 启用内存动态增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:20:12 2020

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import numpy as np
import tensorflow as tf
import os
import argparse
from networks import cos_lr
from models_repo import HADNN1
from call_data import CustomIndoor
from train import runs, runs_pretrained
import time
import pickle

tf.keras.backend.set_floatx('float32')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type=str, default='../data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--warm', type=int, default=5)
    parser.add_argument('--pretrained', type=str, default='False')
    
    args = parser.parse_args()
    
    config = {
        'path_dir': args.path_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'warm': args.warm,
        'pretrained': args.pretrained,
    }
    
    return config

config = parse_args()

np.random.seed(119)
tf.random.set_seed(119)

# Load dataset
dataset = CustomIndoor(config['path_dir'])
n_inputs = dataset.n_rss
n_train_samples = dataset.num_examples
n_test_samples = dataset.num_test_examples
n_hierarchy = 0  # No hierarchy for coordinate-only prediction
hierarchy_info = []  # Empty, as no hierarchy is used

# Prepare data
train_ds = tf.data.Dataset.from_tensor_slices(
    (dataset.train_x.astype(np.float32), dataset.train_c.astype(np.float32))
).shuffle(n_train_samples).batch(config['batch_size'])
test_ds = tf.data.Dataset.from_tensor_slices(
    (dataset.test_x.astype(np.float32), dataset.test_c.astype(np.float32))
).batch(n_test_samples)

# Initialize model
model_name = "HADNN1"
if config['pretrained'] == 'True':
    folder_name = '../data/pretrained/CustomIndoor/' + model_name
    model = tf.keras.models.load_model(folder_name)
else:
    model = HADNN1(n_classes=2, input_dims=dataset.n_rss).build()

# Optimizer and learning rate schedule
optimizer = tf.keras.optimizers.SGD(learning_rate=0., momentum=0.9, nesterov=True, decay=1e-4)
n_train_iter = int(np.ceil(n_train_samples / config['batch_size'] + 1e-10))
length = config['epochs'] * n_train_iter
warm = config['warm'] * n_train_iter
Lr = cos_lr(max_lr=0.01, min_lr=0., warm=warm, end=length)

# Training or evaluation
ss = time.time()
if config['pretrained'] == 'True':
    TRAIN_LOSS, TRAIN_MSE, TRAIN_MSE2, TRAIN_RMSE, TRAIN_MAE, TRAIN_VARIANCE, TRAIN_STD, TEST_LOSS, TEST_MSE, TEST_MSE2, TEST_RMSE, TEST_MAE, TEST_VARIANCE, TEST_STD, train_time, test_time = runs_pretrained(
        model, dataset, train_ds, test_ds, optimizer, Lr, n_hierarchy, hierarchy_info,
        n_train_iter, n_train_samples, hierarchy=False, log_freq=1
    )
    print(f'Training runtime: {train_time:.2f} milliseconds')
    print(f'Testing runtime: {test_time:.2f} milliseconds')
else:
    TRAIN_LOSS, TRAIN_MSE, TRAIN_MSE2, TRAIN_RMSE, TRAIN_MAE, TRAIN_VARIANCE, TRAIN_STD, TEST_LOSS, TEST_MSE, TEST_MSE2, TEST_RMSE, TEST_MAE, TEST_VARIANCE, TEST_STD = runs(
        config["epochs"], model, dataset, train_ds, test_ds, optimizer, Lr, n_hierarchy, hierarchy_info,
        n_train_iter, n_train_samples, hierarchy=False, log_freq=1
    )
ee = time.time()
# Calculate and print runtime in milliseconds
runtime_seconds = ee - ss
runtime_milliseconds = runtime_seconds * 1000
print(f'Algorithm runtime: {runtime_milliseconds:.2f} milliseconds')

# Save model
model.save('C:/Users/Administrator/Desktop/1PAPER/HAD/data/pretrained/CustomIndoor/HADNN1')
# Results
print('#############################################################')
print(f'{model_name} results on CustomIndoor')
print(f'mse on the test set: {TEST_MSE[-1]:.4f}')
print(f'mse2 on the test set: {TEST_MSE2[-1]:.4f}')
print(f'rmse on the test set: {TEST_RMSE[-1]:.4f}')
print(f'mae on the test set: {TEST_MAE[-1]:.4f}')
print(f'variance on the test set: {TEST_VARIANCE[-1]:.4f}')
print(f'std on the test set: {TEST_STD[-1]:.4f}')
print('#############################################################')

Results = {
    'test mse': TEST_MSE[-1],
    'test mse2': TEST_MSE2[-1],
    'test rmse': TEST_RMSE[-1],
    'test mae': TEST_MAE[-1],
    'test variance': TEST_VARIANCE[-1],
    'test std': TEST_STD[-1]
}

with open('../results/result_CustomIndoor_HADNN1.pickle', 'wb') as f:
    pickle.dump(Results, f)