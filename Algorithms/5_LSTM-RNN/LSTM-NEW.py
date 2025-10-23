#############################################################
# 10 locations/ trajectory
# First location is known
# Date: 8 June 2018
#############################################################

import time
import tensorflow
# import lstm
import os
#import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
import math
from pandas import DataFrame
from keras import backend as K

import os
import random
import numpy as np
import tensorflow as tf

# 设置随机种子
def set_random_seed(seed):
    random.seed(seed)  # Python 随机种子
    np.random.seed(seed)  # NumPy 随机种子
    tf.set_random_seed(seed)  # TensorFlow 1.x 随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 确保哈希一致性

set_random_seed(42)

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def err_absolute(y_true, y_pred):
        err = K.sqrt(K.square(y_pred - y_true))
        return err 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

# Open File to write results
File1 = open('Error_Model_5_0_5ms.csv','w')
File2 = open('Traj_Model_5.csv','w') 
# File3 = open('Loss_PredictedL_50k_NoFilter.csv','w') 
# File4 = open('Output_PredictedL_50k_NoFilter.csv','w') 

# NumTotalTraj = 81000
# NumTotalTraj = 252000
NumTotalTraj = 225000

# Configure the sampling set -----------------
epochs_num  = 1000
NumTrajTraining = 20000
NumSam_PerTraj = 1
StartingTraj = 0
StartingSample = StartingTraj*NumSam_PerTraj # the factor of NumSam_PerTraj 
NumSample = NumTrajTraining*NumSam_PerTraj # NumTraj = NumSample/NumSam_PerTraj

# Configure the validation set -----------------
NumTrajVal = 10000
StartingTrajVal = 100000

StartingValidation = StartingTrajVal*NumSam_PerTraj
NumValidation = NumTrajVal*NumSam_PerTraj

# Main Run Thread
global_start_time = time.time()

# input_layer = 6 # X, Y & 11 MAC Addresses
# input_layer = 10
# input_layer = 14
# input_layer = 18
input_layer = 26

# num_rssi_reading = 4
# num_rssi_reading = 8
# num_rssi_reading = 12
# num_rssi_reading = 16
num_rssi_reading = 24


output_layer = 2
timestep = 9
NumBatch = 512
hidden_layer1 = 100
hidden_layer2 = 100

##########################################################
###### Test Trajectory Splitting #########################
########################################################## 
# Read Training data -------5-------------------------------------------
if epochs_num > 0:
    print('> Loading data... ')
    filename = '/home/xiesong/dataset/helipad/WIFI+BLE+SLE/wifi_ble_sle.csv'
    #filename = os.path.join(fileDir, '../Recurrent_Neural_Networks_for_Accurate_RSSI_Indoor_Localization-main/Input_Location_RSSI_10points_365k.csv')
    filename = os.path.abspath(os.path.realpath(filename))
    df1=pd.read_csv(filename)
    df1= np.asarray(df1)
    df1_split = df1[StartingSample:StartingSample+NumSample,:]
    input_training = df1_split.reshape(len(df1_split),timestep,input_layer)
    print(input_training.shape)

    filename = '/home/xiesong/dataset/helipad/WIFI+BLE+SLE/Traj_10points_1000k_5m_s.csv'
    #filename = os.path.join(fileDir, '../Recurrent_Neural_Networks_for_Accurate_RSSI_Indoor_Localization-main/Output_Location_RSSI_10points_365k.csv')
    filename = os.path.abspath(os.path.realpath(filename))
    df2=pd.read_csv(filename)
    df2= np.asarray(df2)
    df2_split = df2[StartingSample:StartingSample+NumSample,:]
    output_training = df2_split.reshape(len(df2_split),timestep,output_layer)
    print(output_training.shape)

    ########################################################################
    # Read Validation data --------------------------------------------------
    #######################################################################
    df3= df1[StartingValidation:StartingValidation+NumValidation,:]
    input_validation = df3.reshape(len(df3),timestep,input_layer)
    print(input_validation.shape)

    df4=df2[StartingValidation:StartingValidation+NumValidation,:]
    output_validation = df4.reshape(len(df4),timestep,output_layer)
    print(output_validation.shape)

####################################################################
# Testing data ------------------------
####################################################################
# RSSI array
filename = '/home/xiesong/dataset/helipad/WIFI+BLE+SLE/test_rssi_wifi_ble_sle.csv'
#filename = os.path.join(fileDir, '../Recurrent_Neural_Networks_for_Accurate_RSSI_Indoor_Localization-main/Long_Traj_6July_v1_AverageFilter.csv')
filename = os.path.abspath(os.path.realpath(filename))
TestData_origin =pd.read_csv(filename)
print(TestData_origin.shape)
TestData_origin = np.asarray(TestData_origin)

LengthTest = len(TestData_origin)
print(len(TestData_origin))
StartTestIdx = 0
StopTestIdx = StartTestIdx + LengthTest+1

# print(TestData.shape)
TestData = TestData_origin[StartTestIdx:StopTestIdx+1, :]
# LengthTest = len(TestData)

# List of Locations 
filename = '/home/xiesong/dataset/helipad/WIFI+BLE+SLE/test_loc.csv'
filename = os.path.abspath(os.path.realpath(filename))
Location_origin = pd.read_csv(filename)
print("Location_origin:", Location_origin)
Location_origin = np.asarray(Location_origin)
Location = Location_origin[StartTestIdx:StopTestIdx+1, :]
print("Location:", Location)

print('> Data Loaded. Compiling...')


####################################################################
# Init 9 first Location ###########################################
####################################################################
L1 = Location[0,:]
L2 = Location[0,:]
L3 = Location[0,:]
L4 = Location[0,:]
L5 = Location[0,:]
L6 = Location[0,:]
L7 = Location[0,:]
L8 = Location[0,:]
L9 = Location[0,:]
L_combine = np.concatenate((L1,L2,L3,L4,L5,L6,L7,L8,L9), axis=0)

# Take RSSI
RSSI_L1 = TestData[0,:]
RSSI_L2 = TestData[1,:] 
RSSI_L3 = TestData[1,:]
RSSI_L4 = TestData[1,:] 
RSSI_L5 = TestData[1,:]
RSSI_L6 = TestData[1,:] 
RSSI_L7 = TestData[1,:]
RSSI_L8 = TestData[1,:] 
RSSI_L9 = TestData[1,:]
RSSI_L10 = TestData[1,:]
RSSI_combine = np.concatenate((RSSI_L2, RSSI_L3, RSSI_L4, RSSI_L5,RSSI_L6, RSSI_L7, RSSI_L8, RSSI_L9, RSSI_L10), axis=0)

# Build the network  --------------------------------------------
model1 = Sequential()
model1.add(LSTM(hidden_layer1, input_shape=(timestep, input_layer), return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(hidden_layer2,return_sequences=True))
model1.add(Dropout(0.2))
model1.add(TimeDistributed(Dense(output_layer)))
model1.summary()
start = time.time()
model1.compile(loss="mse", optimizer="adam",metrics=[rmse])
# model1.compile(loss=rmse, optimizer="adam",metrics=[err_absolute]) #rmse
# root_mean_squared_error
print("> Compilation Time : ", time.time() - start)

# Training --------------------------------------------------------
loss = list()
ResultPlot = DataFrame()

#### Create a copy of input training ##########################
if epochs_num > 0:
    input_training_org = input_training
    input_validation_org = input_validation
###############################################################
# model1.load_weights("new_model.h5")
# 在 model1.load_weights 之前添加
if os.path.exists("new_model.h5"):
    print("Loading pre-trained weights from new_model.h5")
    model1.load_weights("new_model.h5")
else:
    print("No pre-trained weights found. Starting training from scratch.")
StartingValidation = 0
if epochs_num > 0:
    iTimeStep = 1
    iTimeStep_val = 1
    for ep in xrange(epochs_num):
        print("Iteration {} ----- ".format(ep))

        # Fit Model
      #  hist = model1.fit(input_training,output_training, validation_data=(input_validation, output_validation), epochs=1, batch_size=NumBatch, verbose=1)
        hist = model1.fit(input_training, output_training, epochs=1, batch_size=NumBatch, verbose=1, shuffle=False)
        loss.append(hist.history['loss'][0])
      #  File3.write(str(hist.history['loss'][0]) + '\n') # write to file

        # Reform the input training ---------------------------------
   #     print("Training Predict Sample {} ... ".format(iTimeStep))
        Predicted_L_Training = model1.predict(input_training) 
        # get the predicted locations to become the inputs of the next step
        for iSample in xrange(0,len(df1_split)):
            # First location: known
            # Update predicted Location sNormalizeOutputRNN_10points_120k_AverageFiltertarting from the second location
            input_training[iSample,iTimeStep,0] = Predicted_L_Training[iSample,iTimeStep-1,0]
            input_training[iSample,iTimeStep,1] = Predicted_L_Training[iSample,iTimeStep-1,1]
        iTimeStep = iTimeStep + 1 # Update for next time step

        if iTimeStep == timestep:  # reset 1 round -------
            input_training = input_training_org
            iTimeStep = 1
        # --------------------------------------------------------------------------------------

        # Reform the validation set for testing -------------------------------------------------
        # Input is updated by prediction
        # Output: Ideal
    #    print("Validation Predict ...")
    #    Predicted_L_Validation = model1.predict(input_validation) 
    #    for iSample in xrange(0,len(df3)):
    #        input_validation[iSample,iTimeStep_val,0] = Predicted_L_Validation[iSample,iTimeStep_val-1,0]
    #        input_validation[iSample,iTimeStep_val,1] = Predicted_L_Validation[iSample,iTimeStep_val-1,1]
    #    iTimeStep_val = iTimeStep_val + 1        
    #    print("TESTING...........")

     #   if iTimeStep_val == timestep:  # reset 1 round -------
    #        input_validation = input_validation_org
     #       iTimeStep_val = 1
         # ------------------------------------------------------------------------------------------
         # Calculate the validation error -------------------------------------------------


        # model1.fit(input_training,output_training, validation_split=0.2, epochs=epochs_num, batch_size=512)
        model1.save_weights("new_model.h5")
    print('Training duration (s) : ', time.time() - global_start_time)
else:
    print("TESTING...........")
    pass

# Testing --------------------------------------------------------- 
#########################################################
# The first 3 steps: Fill up the buffer
# #######################################################    
# 记录测试开始时间
test_start_time = time.time()
Acc_Location = np.zeros((timestep,2))
Acc_Location[0,:] = L1

for Step in xrange(1,timestep):

    # Update the Locations & RSSIs buffer
    LocationIdx = 0
    RSSIdx = 0
    TestingData = np.zeros(len(L_combine)+len(RSSI_combine))
    for i in xrange(timestep):
        TestingData[LocationIdx] = L_combine[i*2]
        TestingData[LocationIdx+1] = L_combine[i*2+1]
        for j in xrange(len(RSSI_L2)):
            TestingData[j+LocationIdx+2] = (float(RSSI_combine[j+RSSIdx])+100)/100
        LocationIdx = LocationIdx + input_layer
        RSSIdx = RSSIdx + num_rssi_reading

    TestingData = TestingData.reshape(1,timestep,input_layer)
   # print(TestingData)

    # Prediction 
    Predicted_L = model1.predict(TestingData)
    Acc_Location[Step,:] = Predicted_L[0,Step-1,:]
    print(Acc_Location)

    # Update for next step
    # Init 4 first Location
    IdxTemp = 0
    # Update Location
    for j in xrange(Step*2,len(L_combine)):
        L_combine[j] = Acc_Location[Step,IdxTemp]
        if IdxTemp == 0:
            IdxTemp = 1
        else:
            IdxTemp = 0
            pass

    # Take RSSI
    IdxTemp = 0
    for j in xrange(Step*num_rssi_reading,len(RSSI_combine)):
        RSSI_combine[j] = TestData[Step+1,IdxTemp]
        IdxTemp = IdxTemp + 1
        if IdxTemp == num_rssi_reading: # Reach the end
            IdxTemp = 0

#########################################################
# AFter the buffer is full - Do the Test
# #######################################################  

CountArray = np.ones(timestep)
error = np.zeros(LengthTest-1)
# Predicted_array = np.zeros((LengthTest-1,2))
Average_Err = 0

for CountTest in xrange(LengthTest-timestep):
    print("Location {}------------".format(CountTest))
# Update the Locations & RSSIs buffer
    LocationIdx = 0
    RSSIdx = 0
    TestingData = np.zeros(len(L_combine)+len(RSSI_combine))
    for i in xrange(timestep):
        TestingData[LocationIdx] = L_combine[i*2]
        TestingData[LocationIdx+1] = L_combine[i*2+1]
        for j in xrange(len(RSSI_L2)):
            TestingData[j+LocationIdx+2] = (float(RSSI_combine[j+RSSIdx])+100)/100
        LocationIdx = LocationIdx + input_layer
        RSSIdx = RSSIdx + num_rssi_reading

    TestingData = TestingData.reshape(1,timestep,input_layer)
    # print(TestingData)

    # Prediction 
    Predicted_L = model1.predict(TestingData)

   # for t in xrange(timestep): 
   #     File4.write(str(Predicted_L[0,t,0]) + ',') 
   #     File4.write(str(Predicted_L[0,t,1])+ ',') # write to file
   # File4.write('\n')

    # Re-arrange Accumulated Location
    for t in xrange(timestep-1): 
        CountArray[t] = CountArray[t+1]
        Acc_Location[t,:] = Acc_Location[t+1,:]+Predicted_L[0,t,:]
        CountArray[t] = CountArray[t] + 1

    Acc_Location[timestep-1,:] = Predicted_L[0,timestep-1,:] # Update new Location
    CountArray[timestep-1] = 1 #Update New Count
   # print(Acc_Location)
  #  print(CountArray)  
######################################################################
############# UPDATE LOCATION ########################################
######################################################################

    Final_L = Acc_Location[0,:]/CountArray[0]
    CountArray[0] = 1
    # print("Acc_Location:", Acc_Location[0,:])
    # print("CountArray:", CountArray)
    # print("Final_L:", Final_L)
     # Take correct location, compare the result
    Correct_L = Location[CountTest+1,:]
    # print("CountTest:", CountTest, "Correct_L:", Correct_L, "Location[CountTest+1, :]:", Location[CountTest+1, :])
    error[CountTest] = np.sqrt(np.power((Final_L[0] - Correct_L[0]),2)+np.power((Final_L[1] - Correct_L[1]),2))
    print "Predict: {}--- Exact: {} , Error: {}" .format((Final_L[0], Final_L[1]), (Correct_L[0], Correct_L[1]), (error[CountTest]))
    Average_Err = Average_Err + error[CountTest]

    File1.write(str(error[CountTest]) + ' , ') # write to file
    File1.flush()
    File2.write(str(Final_L[0]) + ',') 
    File2.write(str(Final_L[1]) + '\n') # write to file
    # print("Error array after main loop:", error)
    # Re-arrange L_combine
    for t in xrange(timestep-1): 
        L_combine[t*2] = L_combine[(t+1)*2]
        L_combine[t*2+1] = L_combine[(t+1)*2+1] 
    # Update 
    L_combine[(timestep-1)*2] = Predicted_L[0,timestep-1,0] 
    L_combine[(timestep-1)*2+1] = Predicted_L[0,timestep-1,1] 
    # Re-arrange RSSI_combine
    if CountTest+timestep+1 < LengthTest:
        for t in xrange(timestep-1): 
            for k in xrange(num_rssi_reading):
                RSSI_combine[t*num_rssi_reading+k] =  RSSI_combine[(t+1)*num_rssi_reading+k]

        for k in xrange(num_rssi_reading):
            RSSI_combine[(timestep-1)*num_rssi_reading+k] = TestData[CountTest+timestep+1,k]      

###################################################################
############### The Last Locations ###############################
###################################################################
# CountTest = CountTest + 1
# for i in xrange(timestep-1):
#     Final_L = Acc_Location[i+1,:]/CountArray[i+1]
#      # Take correct location, compare the result
#     Correct_L = Location[CountTest+2,:]
    
#     print("CountTest:", CountTest, "Correct_L:", Correct_L, "Location[CountTest+1, :]:", Location[CountTest+1, :])
#     error[CountTest] = np.sqrt(np.power((Final_L[0] - Correct_L[0]),2)+np.power((Final_L[1] - Correct_L[1]),2))
#     print "Predict: {}--- Exact: {} , Error: {}" .format((Final_L[0], Final_L[1]), (Correct_L[0], Correct_L[1]),  error[CountTest])
#     Average_Err = Average_Err + error[CountTest]

#     File1.write(str(error[CountTest]) + ' , ') # write to file
#     print("Error array after main loop:", error)
#     File2.write(str(Final_L[0]) + ',') 
#     File2.write(str(Final_L[1]) + '\n') # write to file
#     CountTest = CountTest + 1
CountTest = LengthTest - timestep  # 确保从 Location 7 开始
for i in range(timestep-1):
    Final_L = Acc_Location[i+1,:]/CountArray[i+1]
    Correct_L = Location[LengthTest-timestep+i+1,:]
    
    error[LengthTest-timestep+i] = np.sqrt(np.power((Final_L[0] - Correct_L[0]),2)+np.power((Final_L[1] - Correct_L[1]),2))
    print "Predict: {}--- Exact: {} , Error: {}" .format((Final_L[0], Final_L[1]), (Correct_L[0], Correct_L[1]), error[LengthTest-timestep+i])
    # print("CountTest:", LengthTest-timestep+i, "Correct_L:", Correct_L, "Location[CountTest+i+1, :]:", Location[LengthTest-timestep+i+1,:])
    Average_Err = Average_Err + error[LengthTest-timestep+i]
    File1.write(str(error[LengthTest-timestep+i]) + ' , ')
    File1.flush()
    File2.write(str(Final_L[0]) + ',')
    File2.write(str(Final_L[1]) + '\n')
    File2.flush()
    # print("Error array after final loop:", error)
print("Processing Location Index:", CountTest+1)
# 计算平均误差和标准偏差
Average_Err = Average_Err / (LengthTest-1)
print("Average Error: ", Average_Err)

Std_Err = 0
for k in range(LengthTest-1):
    Std_Err = Std_Err + np.power((error[k] - Average_Err), 2)
Std_Err = Std_Err / (LengthTest-1)
Std_Err = np.sqrt(Std_Err)
print("Std: ", Std_Err)

# 计算方差
variance = np.mean(np.power(error - Average_Err, 2))
print("Test Variance: ", variance)

# 计算总体的 MSE 和 RMSE
mse = np.mean(np.power(error, 2))
rmse_test = np.sqrt(mse)
print("Test MSE: ", mse)
print("Test RMSE: ", rmse_test)

# 计算测试持续时间
test_duration = time.time() - test_start_time
print("Testing Duration (s): ", test_duration)

# 计算训练持续时间
print("Training Duration (s): ", time.time() - global_start_time)

# Generating ECDF plot and saving data
# Sort errors for ECDF
sorted_errors = np.sort(error)
ecdf = np.arange(1, len(sorted_errors) + 1) / float(len(sorted_errors))

# Save ECDF data to CSV
ecdf_data = np.column_stack((sorted_errors, ecdf))
np.savetxt('ECDF_Data.csv', ecdf_data, delimiter=',', header='Error,ECDF', comments='')

# Create and save ECDF plot
plt.figure()
plt.step(sorted_errors, ecdf, label='ECDF')
plt.xlabel('Localization Error (m)')
plt.ylabel('Cumulative Probability')
plt.title('ECDF of Localization Errors')
plt.grid(True)
plt.legend()
plt.savefig('ECDF_Plot.png')
plt.close()


# 保存结果到文件
# Save results to file with four decimal places
File1.write("Average Error: {:.4f}\n".format(Average_Err))
File1.write("Std: {:.4f}\n".format(Std_Err))
File1.write("Test Variance: {:.4f}\n".format(variance))
File1.write("Test MSE: {:.4f}\n".format(mse))
File1.write("Test RMSE: {:.4f}\n".format(rmse_test))
File1.write("Testing Duration (s): {:.4f}\n".format(test_duration))
File1.write("Training Duration (s): {:.4f}\n".format(time.time() - global_start_time))

# 关闭文件
File1.close()
File2.close()