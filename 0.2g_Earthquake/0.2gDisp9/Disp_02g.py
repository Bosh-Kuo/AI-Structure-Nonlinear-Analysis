# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dropout, Dense
import keras.losses ##
from keras import regularizers
from keras.optimizers import RMSprop, Adam
from keras.layers import LSTM, Activation, CuDNNLSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import time
from random import shuffle
#from sklearn.externals import joblib  # save scaler
import joblib ##
from scipy.stats import norm
import glob
import csv
# Setup GPU for training (use tensorflow v1.9 for CuDNNLSTM)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name) #確定device name
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess) #查看使用中的GPU

# Load training data and validation data
dataDir = '/share/home/bosh/nonlinear_seismic_response_analysis/August/0.2g_Earthquake/'  
resultDir = '/share/home/bosh/nonlinear_seismic_response_analysis/August/0.2g_Earthquake/Disp/0.2gDisp9/'  
mat_data = scipy.io.loadmat(dataDir+'NewData_Disp.mat')
mat_analysis = scipy.io.loadmat(dataDir+'Disp02g_Analysis.mat')

# special index
TrainGoodInd = mat_analysis['TrainGoodInd'][8][0][0]
TestGoodInd = mat_analysis['TestGoodInd'][8][0][0]

# original data
X_train_stack = mat_data['X_train_stack'][TrainGoodInd]
y_Disp_train_stack = mat_data['y_Disp_train_stack'][TrainGoodInd]
X_test_stack = mat_data['X_test_stack'][TestGoodInd]
y_Disp_test_stack = mat_data['y_Disp_test_stack'][TestGoodInd]

def ScalingInput(train,test):
    train_flatten = np.reshape(train, [train.shape[0]*train.shape[1]*train.shape[2],1])  
    test_flatten = np.reshape(test, [test.shape[0]*test.shape[1]*test.shape[2],1]) 
    scaler_X = MinMaxScaler(feature_range=(-1, 1))  #標準化器
    scaler_X.fit(train_flatten)  #對應到要縮放的資料
    train_flatten_map = scaler_X.transform(train_flatten)  
    train_map = np.reshape(train_flatten_map, [train.shape[0], train.shape[1], train.shape[2]]) 
    test_flatten_map = scaler_X.transform(test_flatten)  
    test_map = np.reshape(test_flatten_map, [test.shape[0], test.shape[1], test.shape[2]]) 
    return train_map, test_map, scaler_X
 
def ScalingOutput(train,test):
    train_flatten = np.reshape(train, [train.shape[0]*train.shape[1] ,train.shape[2]])  
    test_flatten = np.reshape(test, [test.shape[0]*test.shape[1], test.shape[2]]) 
    scaler_y = MinMaxScaler(feature_range=(-1, 1))  #標準化器
    scaler_y.fit(train_flatten)  #對應到要縮放的資料
    train_flatten_map = scaler_y.transform(train_flatten)  
    train_map = np.reshape(train_flatten_map, [train.shape[0], train.shape[1], train.shape[2]]) 
    test_flatten_map = scaler_y.transform(test_flatten)  
    test_map = np.reshape(test_flatten_map, [test.shape[0], test.shape[1], test.shape[2]]) 
    return train_map, test_map, scaler_y

X_train_scale_stack, X_test_scale_stack, scaler_X = ScalingInput(X_train_stack,X_test_stack)
y_Disp_train_scale_stack, y_Disp_test_scale_stack, scaler_y_Disp = ScalingOutput(y_Disp_train_stack,y_Disp_test_stack)

# Save scaler
joblib.dump(scaler_X, resultDir+ 'scaler_X.save')
joblib.dump(scaler_y_Disp, resultDir+ 'scaler_y_Disp.save')

#-------------------------------------------------------------------------------
# Building model
data_dim = X_train_scale_stack.shape[2]  # number of input features
timesteps = X_train_scale_stack.shape[1]
num_classes = y_Disp_train_scale_stack.shape[2]  # number of output features
batch_size = 32
rms = RMSprop(lr=0.001, decay=0.0001)
adam = Adam(lr=0.001, decay=0.0001)

#-------------------------------------------------------------------------------
model = Sequential()
model.add(CuDNNLSTM(100, return_sequences=True, stateful=False, input_shape=(None, data_dim), recurrent_regularizer=regularizers.l2(0.001)))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.summary()


model.compile(loss='mean_squared_error',  # categorical_crossentropy, mean_squared_error, mean_absolute_error
              optimizer=adam,  # RMSprop(), Adagrad, Nadam, Adagrad, Adadelta, Adam, Adamax,
              metrics=['mse'])
best_loss = 100
train_loss = []
validation_loss = []
history = []

#-------------------------------------------------------------------------------
# Training
with tf.device('/device:GPU:0'):  #指定session在GPU:0上運行，在colab上也不需要特別指定
    #限制GPU使用率
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  #動態申請顯存
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4  #限制GPU使用率為40%
    session = tf.Session(config=config)
    # tf.Session(config=tf.ConfigProto(log_device_placement=True))

#-------------------------------------------------------------------------------
    start = time.time()
    epochs = 1500  
    Ind = list(range(len(X_train_scale_stack)))
    shuffle(Ind)
    ratio_split = 0.8
    Ind_train = Ind[0:round(ratio_split*len(X_train_scale_stack))]
    Ind_validation = Ind[round(ratio_split*len(X_train_scale_stack)):]
    X_train = X_train_scale_stack[Ind_train]
    y_train = y_Disp_train_scale_stack[Ind_train]
    X_validation = X_train_scale_stack[Ind_validation]
    y_validation = y_Disp_train_scale_stack[Ind_validation]
    for e in range(epochs):
        print('epoch = ', e + 1)
#-------------------------------------------------------------------------------
        model.fit(X_train, y_train,
                  batch_size=batch_size,
                  # validation_split=0.3,
                  validation_data=(X_validation, y_validation),
                  shuffle=True,
                  epochs=1)
        score0 = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
        score = model.evaluate(X_validation, y_validation, batch_size=batch_size, verbose=0)
        train_loss.append(score0[0]) 
        validation_loss.append(score[0])

        # Save the best trained model with minimum testing loss
        if validation_loss[e] <= best_loss:
            best_epoch = e
            best_loss = validation_loss[e]
            model.save(resultDir+'best_model.h5')
#-------------------------------------------------------------------------------
    model.save(resultDir+'1500epochs_model.h5')    
    end = time.time()
    running_time = (end - start)/3600
    print('Running Time: ', running_time, ' hour')
    print('best_epoch: ', best_epoch)

    
#-------------------------------------------------------------------------------
# Plot training and validation loss
plt.figure()
plt.plot(np.array(train_loss), 'b-',label='Training Loss')
plt.plot(np.array(validation_loss), 'm-',label='Validation Loss')
plt.title('(0.2g data)Displacement model loss curve')
plt.xlabel('epoch')
plt.ylabel('MSE Loss')
plt.legend()
ax = plt.gca()
ax.set_yscale('log')
plt.savefig(resultDir+'Disp02g_loss_curve(log).png')    

plt.figure()
plt.plot(np.array(train_loss), 'b-',label='Training Loss')
plt.plot(np.array(validation_loss), 'm-',label='Validation Loss')
plt.title('(0.2g data)Displacement model loss curve')
plt.xlabel('epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(resultDir+'Disp02g_loss_curve(linear).png')    

#-------------------------------------------------------------------------------
# 儲存資料與訓練結果
scipy.io.savemat(resultDir+'Disp02g_reult.mat',
                 { 'train_loss': train_loss,'validation_loss':validation_loss,'best_loss':best_loss})

#-------------------------------------------------------------------------------
# Load the best model
def prediction(X_scale_stack,y_scale_stack):
      pred = model.predict(X_scale_stack) 
      pred_flatten = np.reshape(pred, [pred.shape[0]*pred.shape[1], pred.shape[2]])  
      pred = scaler_y_Disp.inverse_transform(pred_flatten)  #將預測數值縮放回原大小 
      pred = np.reshape(pred, [y_scale_stack.shape[0], y_scale_stack.shape[1], y_scale_stack.shape[2]])  
      return pred
  
def MSE(Response_true,Response_pred):
      ErrorList=[]
      for i in range(Response_true.shape[0]):
          for k in range(Response_true.shape[2]):
            true_std = Response_true[i,:,k] / (Response_true[i,:,k].max() - Response_true[i,:,k].min())
            true_scaled = true_std * (1-(-1)) + (-1)
            pred_std = Response_pred[i,:,k] / (Response_true[i,:,k].max() - Response_true[i,:,k].min())
            pred_scaled = pred_std * (1-(-1)) + (-1)
    
            ErrorList.append( sum((pred_scaled- true_scaled)**2)/Response_true.shape[1] ) 
      Mse = np.array(ErrorList).reshape(Response_true.shape[0],Response_true.shape[2])
      return Mse
    
ModelList = ['best_model','1500epochs_model']
for Model in ModelList:
    model = load_model(resultDir + Model + '.h5')
        
    # Predict Training data and Testing data  
    y_train_stack_pred = prediction(X_train_scale_stack,y_Disp_train_scale_stack)
    y_test_stack_pred = prediction(X_test_scale_stack, y_Disp_test_scale_stack)    
    #-------------------------------------------------------------------------------
    # normalized MSE
    Train_MSE = MSE(y_Disp_train_stack,y_train_stack_pred)
    Test_MSE = MSE(y_Disp_test_stack,y_test_stack_pred)
    
    #-------------------------------------------------------------------------------
    # 繪製MSE分布
    plt.figure()
    for i in range(1,10):
      plt.scatter((np.ones(Train_MSE.shape[0])+i).astype(np.int8),Train_MSE[:,i-1],s=30,c='red',marker='o',alpha=0.5,)
    plt.title('Training Data MSE (0.2g-Displacement) '+Model)
    plt.xlabel('F')
    plt.ylabel('MSE')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(linestyle='--')
    plt.savefig(resultDir+Model+' Train_MSE.png')    
    
    plt.figure()
    for i in range(1,10):
      plt.scatter((np.ones(Test_MSE.shape[0])+i).astype(np.int8),Test_MSE[:,i-1],s=30,c='black',marker='^',alpha=0.5,)
    plt.title('Testing Data MSE (0.2g-Displacement) '+Model)
    plt.xlabel('F')
    plt.ylabel('MSE')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.grid(linestyle='--')
    plt.savefig(resultDir+Model+' Test_MSE.png')    
    
    #-------------------------------------------------------------------------------
    # 繪製MSE百分比圖
    PerformanceIndex = 0.005
    MSEList = [[Train_MSE,'(0.2g Displacement data)Training'],[Test_MSE,'(0.2g Displacement data)Testing']]
    for sample in MSEList:
      GoodList = []
      BadList = []
      for i in range(9):
        StoryMSE = sample[0][:,i]
        GoodData = StoryMSE[StoryMSE<=PerformanceIndex]
        GoodList.append(GoodData.shape[0]/StoryMSE.shape[0]*100)
        BadList.append(100 - GoodData.shape[0] / StoryMSE.shape[0]*100)
    
      x = np.arange(9)+2
      plt.figure()
      plt.bar(x, GoodList, label='predict well')
      plt.bar(x, BadList, bottom=GoodList,label='predict bad')
      for xy in zip(x, GoodList):
        plt.annotate("%.1f" %xy[1]+'%', xy=xy, xytext=(-15, 2), textcoords='offset points')
      plt.title(Model+sample[1]+' Data Prediction Result')
      plt.xlabel('F')
      plt.ylabel('Prediction Well Percentage(%)')
      plt.xticks(x)
      plt.yticks(np.arange(0, 105, 10))
      plt.legend(loc = 'lower left') 
      plt.savefig(resultDir+Model+sample[1]+'Percentage.png')    



