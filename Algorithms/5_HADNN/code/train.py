#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 20:44:01 2022

@author: jaehooncha

@email: chajaehoon79@gmail.com
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from networks import mapper_superclass
import time

compute_loss = tf.keras.losses.MeanSquaredError()

def train_step(inputs, model, optimizer, dataset):
    X, Y = inputs
    with tf.GradientTape() as tape:
        logit = model(X)
        loss = compute_loss(Y, logit)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Denormalize predictions and true values
    recover_logit = logit.numpy()
    recover_true = Y.numpy()
    recover_logit[:, 0] = recover_logit[:, 0] * dataset.x_std + dataset.x_mean
    recover_logit[:, 1] = recover_logit[:, 1] * dataset.y_std + dataset.y_mean
    recover_true[:, 0] = recover_true[:, 0] * dataset.x_std + dataset.x_mean
    recover_true[:, 1] = recover_true[:, 1] * dataset.y_std + dataset.y_mean
    
    # Calculate Euclidean distance (original MSE)
    dist = np.linalg.norm(recover_logit - recover_true, axis=-1)
    mse = np.mean(dist)
    
    # Calculate MSE2 (traditional Mean Squared Error)
    mse2 = np.mean(np.sum((recover_logit - recover_true) ** 2, axis=-1))
    
    # Calculate RMSE
    rmse = np.sqrt(mse2)
    
    # Calculate MAE
    mae = np.mean(np.sum(np.abs(recover_logit - recover_true), axis=-1))
    
    return loss, [loss], [logit], mse, mse2, rmse, mae, dist

def test_step(inputs, model, dataset):
    X, Y = inputs
    logit = model(X)
    loss = compute_loss(Y, logit)
    
    # Denormalize predictions and true values
    recover_logit = logit.numpy()
    recover_true = Y.numpy()
    recover_logit[:, 0] = recover_logit[:, 0] * dataset.x_std + dataset.x_mean
    recover_logit[:, 1] = recover_logit[:, 1] * dataset.y_std + dataset.y_mean
    recover_true[:, 0] = recover_true[:, 0] * dataset.x_std + dataset.x_mean
    recover_true[:, 1] = recover_true[:, 1] * dataset.y_std + dataset.y_mean
    
    # Calculate Euclidean distance (original MSE)
    dist = np.linalg.norm(recover_logit - recover_true, axis=-1)
    mse = np.mean(dist)
    
    # Calculate MSE2 (traditional Mean Squared Error)
    mse2 = np.mean(np.sum((recover_logit - recover_true) ** 2, axis=-1))
    
    # Calculate RMSE
    rmse = np.sqrt(mse2)
    
    # Calculate MAE
    mae = np.mean(np.sum(np.abs(recover_logit - recover_true), axis=-1))
    
    return loss, [loss], [logit], mse, mse2, rmse, mae, dist

def plot_ecdf(distances, filename, title="测试集欧几里得距离误差的ECDF"):
    """
    Plot the ECDF of the given distances as a step curve, save to filename,
    and export individual Euclidean distance errors to a CSV file.
    """
    # Sort distances and compute ECDF
    sorted_distances = np.sort(distances)
    n = len(sorted_distances)
    y = np.arange(1, n + 1) / n
    
    # Create step plot
    plt.figure(figsize=(8, 6))
    plt.step(sorted_distances, y, where='post', label='ECDF')
    plt.xlabel('欧几里得距离误差')
    plt.ylabel('累积概率')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export individual Euclidean distance errors to CSV
    csv_filename = filename.replace('.png', '.csv')
    ecdf_data = pd.DataFrame({
        'Euclidean_Distance_Error': distances
    })
    ecdf_data.to_csv(csv_filename, index=False)

def runs(epochs, model, dataset, train_ds, test_ds, optimizer, Lr, n_hierarchy, hierarchy_info, n_train_iter, n_train_samples, hierarchy=False, log_freq=1):
    k = 0
    TRAIN_LOSS = []
    TRAIN_MSE = []
    TRAIN_MSE2 = []
    TRAIN_RMSE = []
    TRAIN_MAE = []
    TRAIN_VARIANCE = []
    TRAIN_STD = []
    TEST_LOSS = []
    TEST_MSE = []
    TEST_MSE2 = []
    TEST_RMSE = []
    TEST_MAE = []
    TEST_VARIANCE = []
    TEST_STD = []
    last_test_dists = []  # Store test distances from the last epoch
    
    for epoch in range(epochs):
        train_loss = np.zeros(shape=n_hierarchy + 1)
        train_mse = 0.0
        train_mse2 = 0.0
        train_rmse = 0.0
        train_mae = 0.0
        train_dists = []  # Collect distances for variance calculation
        
        for epoch_set in train_ds:
            inputs = [epoch_set[0], epoch_set[1]]
            tf.keras.backend.set_value(optimizer.lr, Lr[k])
            loss, losses, logits, mse, mse2, rmse, mae, dist = train_step(inputs, model, optimizer, dataset)
            train_loss = [train_loss[j] + losses[j].numpy() for j in range(n_hierarchy + 1)]
            train_mse += mse
            train_mse2 += mse2
            train_rmse += rmse
            train_mae += mae
            train_dists.extend(dist)
            k += 1
        
        train_loss = np.array(train_loss) / n_train_iter
        train_mse /= n_train_iter
        train_mse2 /= n_train_iter
        train_rmse /= n_train_iter
        train_mae /= n_train_iter
        # Calculate variance and std of Euclidean distances
        train_variance = np.var(train_dists)
        train_std = np.sqrt(train_variance)
        
        TRAIN_LOSS.append(train_loss)
        TRAIN_MSE.append(train_mse)
        TRAIN_MSE2.append(train_mse2)
        TRAIN_RMSE.append(train_rmse)
        TRAIN_MAE.append(train_mae)
        TRAIN_VARIANCE.append(train_variance)
        TRAIN_STD.append(train_std)
        
        test_loss = np.zeros(shape=n_hierarchy + 1)
        test_mse = 0.0
        test_mse2 = 0.0
        test_rmse = 0.0
        test_mae = 0.0
        test_dists = []  # Collect distances for variance calculation and ECDF
        
        for epoch_set in test_ds:
            inputs = [epoch_set[0], epoch_set[1]]
            loss, losses, logits, mse, mse2, rmse, mae, dist = test_step(inputs, model, dataset)
            test_loss = [test_loss[j] + losses[j].numpy() for j in range(n_hierarchy + 1)]
            test_mse += mse
            test_mse2 += mse2
            test_rmse += rmse
            test_mae += mae
            test_dists.extend(dist)
        
        test_loss = np.array(test_loss) / 1  # Single batch for test
        test_mse /= 1
        test_mse2 /= 1
        test_rmse /= 1
        test_mae /= 1
        # Calculate variance and std of Euclidean distances
        test_variance = np.var(test_dists)
        test_std = np.sqrt(test_variance)
        
        # Store test_dists from the last epoch
        if epoch == epochs - 1:
            last_test_dists = test_dists
        
        TEST_LOSS.append(test_loss)
        TEST_MSE.append(test_mse)
        TEST_MSE2.append(test_mse2)
        TEST_RMSE.append(test_rmse)
        TEST_MAE.append(test_mae)
        TEST_VARIANCE.append(test_variance)
        TEST_STD.append(test_std)
        
        template = 'epoch: {}\n'.format(epoch + 1)
        template += ' train loss: {}\n'.format(train_loss[0])
        template += ' train mse: {}\n'.format(train_mse)
        template += ' train mse2: {}\n'.format(train_mse2)
        template += ' train rmse: {}\n'.format(train_rmse)
        template += ' train mae: {}\n'.format(train_mae)
        template += ' train variance: {}\n'.format(train_variance)
        template += ' train std: {}\n'.format(train_std)
        template += ' test loss: {}\n'.format(test_loss[0])
        template += ' test mse: {}\n'.format(test_mse)
        template += ' test mse2: {}\n'.format(test_mse2)
        template += ' test rmse: {}\n'.format(test_rmse)
        template += ' test mae: {}\n'.format(test_mae)
        template += ' test variance: {}\n'.format(test_variance)
        template += ' test std: {}\n'.format(test_std)
        
        if (epoch + 1) % log_freq == 0:
            print(template)
    
    # Plot ECDF and export data for test distances from the last epoch
    plot_ecdf(last_test_dists, './ecdf_test_final.png', 
              title='测试集欧几里得距离误差的ECDF (最终模型)')
    
    return TRAIN_LOSS, TRAIN_MSE, TRAIN_MSE2, TRAIN_RMSE, TRAIN_MAE, TRAIN_VARIANCE, TRAIN_STD, TEST_LOSS, TEST_MSE, TEST_MSE2, TEST_RMSE, TEST_MAE, TEST_VARIANCE, TEST_STD

def runs_pretrained(model, dataset, train_ds, test_ds, optimizer, Lr, n_hierarchy, hierarchy_info, n_train_iter, n_train_samples, hierarchy=False, log_freq=1):
    k = 0
    TRAIN_LOSS = []
    TRAIN_MSE = []
    TRAIN_MSE2 = []
    TRAIN_RMSE = []
    TRAIN_MAE = []
    TRAIN_VARIANCE = []
    TRAIN_STD = []
    TEST_LOSS = []
    TEST_MSE = []
    TEST_MSE2 = []
    TEST_RMSE = []
    TEST_MAE = []
    TEST_VARIANCE = []
    TEST_STD = []
    total_train_time = 0.0  # Track training (evaluation) time
    total_test_time = 0.0   # Track testing time
    
    train_loss = np.zeros(shape=n_hierarchy + 1)
    train_mse = 0.0
    train_mse2 = 0.0
    train_rmse = 0.0
    train_mae = 0.0
    train_dists = []  # Collect distances for variance calculation
    
    # Training phase (evaluation only) with timing
    train_start = time.time()
    for epoch_set in train_ds:
        inputs = [epoch_set[0], epoch_set[1]]
        tf.keras.backend.set_value(optimizer.lr, Lr[k])
        loss, losses, logits, mse, mse2, rmse, mae, dist = test_step(inputs, model, dataset)
        train_loss = [train_loss[j] + losses[j].numpy() for j in range(n_hierarchy + 1)]
        train_mse += mse
        train_mse2 += mse2
        train_rmse += rmse
        train_mae += mae
        train_dists.extend(dist)
        k += 1
    train_end = time.time()
    total_train_time = (train_end - train_start) * 1000  # Convert to milliseconds
    
    train_loss = np.array(train_loss) / n_train_iter
    train_mse /= n_train_iter
    train_mse2 /= n_train_iter
    train_rmse /= n_train_iter
    train_mae /= n_train_iter
    # Calculate variance and std of Euclidean distances
    train_variance = np.var(train_dists)
    train_std = np.sqrt(train_variance)
    
    TRAIN_LOSS.append(train_loss)
    TRAIN_MSE.append(train_mse)
    TRAIN_MSE2.append(train_mse2)
    TRAIN_RMSE.append(train_rmse)
    TRAIN_MAE.append(train_mae)
    TRAIN_VARIANCE.append(train_variance)
    TRAIN_STD.append(train_std)
    
    test_loss = np.zeros(shape=n_hierarchy + 1)
    test_mse = 0.0
    test_mse2 = 0.0
    test_rmse = 0.0
    test_mae = 0.0
    test_dists = []  # Collect distances for variance calculation and ECDF
    
    # Testing phase with timing
    test_start = time.time()
    for epoch_set in test_ds:
        inputs = [epoch_set[0], epoch_set[1]]
        loss, losses, logits, mse, mse2, rmse, mae, dist = test_step(inputs, model, dataset)
        test_loss = [test_loss[j] + losses[j].numpy() for j in range(n_hierarchy + 1)]
        test_mse += mse
        test_mse2 += mse2
        test_rmse += rmse
        test_mae += mae
        test_dists.extend(dist)
    test_end = time.time()
    total_test_time = (test_end - test_start) * 1000  # Convert to milliseconds
    
    test_loss = np.array(test_loss) / 1  # Single batch for test
    test_mse /= 1
    test_mse2 /= 1
    test_rmse /= 1
    test_mae /= 1
    # Calculate variance and std of Euclidean distances
    test_variance = np.var(test_dists)
    test_std = np.sqrt(test_variance)
    
    # Plot ECDF and export data for test distances
    plot_ecdf(test_dists, './ecdf_test_pretrained.png', 
              title='预训练模型测试集欧几里得距离误差的ECDF')
    
    TEST_LOSS.append(test_loss)
    TEST_MSE.append(test_mse)
    TEST_MSE2.append(test_mse2)
    TEST_RMSE.append(test_rmse)
    TEST_MAE.append(test_mae)
    TEST_VARIANCE.append(test_variance)
    TEST_STD.append(test_std)
    
    template = 'Pretrained model evaluation\n'
    template += ' train loss: {}\n'.format(train_loss[0])
    template += ' train mse: {}\n'.format(train_mse)
    template += ' train mse2: {}\n'.format(train_mse2)
    template += ' train rmse: {}\n'.format(train_rmse)
    template += ' train mae: {}\n'.format(train_mae)
    template += ' train variance: {}\n'.format(train_variance)
    template += ' train std: {}\n'.format(train_std)
    template += ' test loss: {}\n'.format(test_loss[0])
    template += ' test mse: {}\n'.format(test_mse)
    template += ' test mse2: {}\n'.format(test_mse2)
    template += ' test rmse: {}\n'.format(test_rmse)
    template += ' test mae: {}\n'.format(test_mae)
    template += ' test variance: {}\n'.format(test_variance)
    template += ' test std: {}\n'.format(test_std)
    
    print(template)
    
    return TRAIN_LOSS, TRAIN_MSE, TRAIN_MSE2, TRAIN_RMSE, TRAIN_MAE, TRAIN_VARIANCE, TRAIN_STD, TEST_LOSS, TEST_MSE, TEST_MSE2, TEST_RMSE, TEST_MAE, TEST_VARIANCE, TEST_STD, total_train_time, total_test_time