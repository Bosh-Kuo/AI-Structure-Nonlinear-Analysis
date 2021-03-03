import os
import glob
import csv
import scipy.io
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib  # save scaler

#------------------------------------------------------------------------------
#整理地表加速度
def GroundAccelerationToArray(folders):
    earthquakedir = glob.glob(folders[0]+'/*')
    earthquakedir.sort()
    print(earthquakedir[0]) #check
    GroundAcceleration = np.loadtxt(earthquakedir[0],delimiter=',',usecols=(0,1))  #寫入地震csv檔,用垂直列0,1  #注意earthquakedir為list
    steps = [GroundAcceleration[1,0]-GroundAcceleration[0,0]]
    GroundAcceleration = np.pad(GroundAcceleration[:,1], (0,21001-GroundAcceleration.shape[0]), mode = 'constant').reshape(1,21001)   #補零至21001
    
    for i in range(1,len(folders)):
        earthquakedir = glob.glob(folders[i]+'/*')
        earthquakedir.sort()
        print(earthquakedir[0]) #check
        earthquake = np.loadtxt(earthquakedir[0],delimiter=',',usecols=(0,1))  #寫入地震csv檔,只用1垂直列
        step = earthquake[1,0]-earthquake[0,0]
        earthquake = np.pad(earthquake[:,1],(0,21001-earthquake.shape[0]),mode = 'constant').reshape(1,21001)  #補零至21001
        GroundAcceleration = np.append(GroundAcceleration,earthquake, axis=0)
        steps.append(step)
    steps = np.array(steps)  
    print('分隔線-----------------------------')   
    return steps,GroundAcceleration  #回傳時間間隔array與地表加速度array

#------------------------------------------------------------------------------
#整理結構反應
def ResponseToArray(folders,ResponseType):
    Response = np.loadtxt(folders[0]+ResponseType, delimiter=',',usecols=(1,2,3,4,5,6,7,8,9),skiprows=(1)).reshape(1,21001,9)  #讀取地震csv檔
    for i in range(1,len(folders)):
        response = np.loadtxt(folders[i]+ResponseType, delimiter=',',usecols=(1,2,3,4,5,6,7,8,9),skiprows=(1)).reshape(1,21001,9)  #讀取地震csv檔
        Response = np.append(Response,response, axis=0)
    return Response  #回傳結構反應array

#-------------------------------------------------------------------------------
# Process full sequence to stacked sequence
def Generate_data(X_data0, y_data0,y_data1,y_data2, window_size=10):
    X_new_temp = []
    y_new_temp = []
    y_new_temp1 = []
    y_new_temp2 = []
    for ii in range(len(X_data0)):
        X_temp = X_data0[ii]
        y_temp = y_data0[ii]
        y_temp1 = y_data1[ii]
        y_temp2 = y_data2[ii]
        X_new = []
        y_new = []
        y_new1 = []
        y_new2 = []
        for jj in range(int(np.floor(len(X_temp) / window_size))):
            X_new.append(X_temp[jj * window_size:(jj + 1) * window_size])
            y_new.append(y_temp[(jj + 1) * window_size - 1, :])
            y_new1.append(y_temp1[(jj + 1) * window_size - 1, :])
            y_new2.append(y_temp2[(jj + 1) * window_size - 1, :])
            # y_new.append(y_temp[(jj + 1) * window_size - 1])
        X_new_temp.append(np.array(X_new))
        y_new_temp.append(np.array(y_new))
        y_new_temp1.append(np.array(y_new1))
        y_new_temp2.append(np.array(y_new2))
    X_data_new0 = np.array(X_new_temp)
    y_data_new0 = np.array(y_new_temp)
    y_data_new1 = np.array(y_new_temp1)
    y_data_new2 = np.array(y_new_temp2)

    return X_data_new0, y_data_new0, y_data_new1, y_data_new2

def shuffleIndex(GroundAcceleration):
    Index = np.arange(GroundAcceleration.shape[0])
    np.random.shuffle(Index)
    trainInd = Index[:int(np.floor(GroundAcceleration.shape[0]*0.7))]
    trainInd.sort()
    testInd = Index[int(np.floor(GroundAcceleration.shape[0]*0.7)):]
    testInd.sort()
    return trainInd,testInd ##回傳打亂後要作為training data的index與要作為testing data的index


path_02g = r'/share/home/bosh/nonlinear_seismic_response_analysis/August/0.2g_Earhquake' #目標資料夾
NObase_source_path_02g = r'/share/home/nonlinear_analysis_dataset/dataset_no_baseline_correction/0.2g_ml_dataset'  #914資料夾
base_source_path_02g = r'/share/home/nonlinear_analysis_dataset/dataset_baseline_correction/0.2g_ml_dataset'  #718資料夾

path_03g = r'/share/home/bosh/nonlinear_seismic_response_analysis/August/0.3g_Earhquake' #目標資料夾
NObase_source_path_03g = r'/share/home/nonlinear_analysis_dataset/dataset_no_baseline_correction/0.3g_ml_dataset'  #655資料夾
base_source_path_03g = r'/share/home/nonlinear_analysis_dataset/dataset_baseline_correction/0.3g_ml_dataset'  #485資料夾

path_04g = r'/share/home/bosh/nonlinear_seismic_response_analysis/August/0.4g_Earhquake' #目標資料夾
NObase_source_path_04g = r'/share/home/nonlinear_analysis_dataset/dataset_no_baseline_correction/0.4g_ml_dataset'  #518資料夾
base_source_path_04g = r'/share/home/nonlinear_analysis_dataset/dataset_baseline_correction/0.4g_ml_dataset'  #343資料夾
List = [[path_02g,NObase_source_path_02g,base_source_path_02g],[path_03g,NObase_source_path_03g,base_source_path_03g],[path_04g,NObase_source_path_04g,base_source_path_04g]]
ResponseType = ['/NodeAccAbs.csv','/NodeDis.csv','/NodeStoryDriftRatio.csv']

for level in List:
    NObasefolders = glob.glob(level[1]+'/*')  #914/655/518個0.2/0.3g/0.4g NonBC地震資料夾路徑字串list
    NObasefolders.sort()  #將資料夾按順序排列
    basefolders = glob.glob(level[2]+'/*')  #718/485/343個.2/0.3g/0.4g BC地震資料夾路徑字串list
    basefolders.sort()  #將資料夾按順序排列
    
    #-------------------------------------------------------------------------------
    # 無基線修正(1)
    step1, GroundAcceleration1 = GroundAccelerationToArray(NObasefolders)
    Acc1 = ResponseToArray(NObasefolders,ResponseType[0])
    Disp1 = ResponseToArray(NObasefolders,ResponseType[1])
    StoryDrift1 = ResponseToArray(NObasefolders,ResponseType[2])
    name1 = [file for file in os.listdir(level[1])]
    name1.sort()
    name1 = np.array(name1)
    trainInd1,testInd1 = shuffleIndex(GroundAcceleration1)
    
    #地表加速度
    GroundAcceleration1_train = GroundAcceleration1[trainInd1]
    GroundAcceleration1_test = GroundAcceleration1[testInd1]
    #結構反應
    Acc1_train = Acc1[trainInd1]
    Disp1_train = Disp1[trainInd1]
    StoryDrift1_train = StoryDrift1[trainInd1]
    
    Acc1_test = Acc1[testInd1]
    Disp1_test = Disp1[testInd1]
    StoryDrift1_test = StoryDrift1[testInd1]
    #地震名與timestep
    name1_train = name1[trainInd1]
    name1_test = name1[testInd1]
    step1_train = step1[trainInd1]
    step1_test = step1[testInd1]

    #-------------------------------------------------------------------------------
    # 基線修正(2)
    step2, GroundAcceleration2 = GroundAccelerationToArray(basefolders)
    Acc2 = ResponseToArray(basefolders,ResponseType[0])
    Disp2 = ResponseToArray(basefolders,ResponseType[1])
    StoryDrift2 = ResponseToArray(basefolders,ResponseType[2])
    name2 = [file for file in os.listdir(level[2])]
    name2.sort()
    name2 = np.array(name2)
    trainInd2,testInd2 = shuffleIndex(GroundAcceleration2)
    
    #地表加速度
    GroundAcceleration2_train = GroundAcceleration2[trainInd2]
    GroundAcceleration2_test = GroundAcceleration2[testInd2]
    #結構反應
    Acc2_train = Acc2[trainInd2]
    Disp2_train = Disp2[trainInd2]
    StoryDrift2_train = StoryDrift2[trainInd2]
    
    Acc2_test = Acc2[testInd2]
    Disp2_test = Disp2[testInd2]
    StoryDrift2_test = StoryDrift2[testInd2]
    #地震名與timestep
    name2_train = name2[trainInd2]
    name2_test = name2[testInd2]
    step2_train = step2[trainInd2]
    step2_test = step2[testInd2]
    
    #-------------------------------------------------------------------------------
    # 合併(Non BC + BC)
    name_train = np.concatenate((name1_train, name2_train), axis=0)
    name_test = np.concatenate((name1_test, name2_test), axis=0)
    
    step_train = np.concatenate((step1_train, step2_train), axis=0)
    step_test = np.concatenate((step1_test, step2_test), axis=0)
    
    GroundAcceleration_train = np.concatenate((GroundAcceleration1_train, GroundAcceleration2_train), axis=0)
    GroundAcceleration_test = np.concatenate((GroundAcceleration1_test, GroundAcceleration2_test), axis=0)
    
    Acc_train = np.concatenate((Acc1_train, Acc2_train), axis=0)
    Acc_test = np.concatenate((Acc1_test, Acc2_test), axis=0)
    
    Disp_train = np.concatenate((Disp1_train, Disp2_train), axis=0)
    Disp_test = np.concatenate((Disp1_test, Disp2_test), axis=0)
    
    StoryDrift_train = np.concatenate((StoryDrift1_train, StoryDrift2_train), axis=0)
    StoryDrift_test = np.concatenate((StoryDrift1_test, StoryDrift2_test), axis=0)
    
    #-------------------------------------------------------------------------------
    # stack data
    # original
    GroundAcceleration_train = np.reshape(GroundAcceleration_train,[GroundAcceleration_train.shape[0],GroundAcceleration_train.shape[1],1])
    X_train_stack, y_Acc_train_stack, y_Disp_train_stack, y_StoryDrift_train_stack = Generate_data(GroundAcceleration_train, Acc_train, Disp_train, StoryDrift_train) 
    X_train_stack = np.reshape(X_train_stack, [X_train_stack.shape[0], X_train_stack.shape[1], X_train_stack.shape[2]])  
    
    GroundAcceleration_test = np.reshape(GroundAcceleration_test,[GroundAcceleration_test.shape[0],GroundAcceleration_test.shape[1],1])
    X_test_stack, y_Acc_test_stack, y_Disp_test_stack, y_StoryDrift_test_stack = Generate_data(GroundAcceleration_test, Acc_test, Disp_test, StoryDrift_test) 
    X_test_stack = np.reshape(X_test_stack, [X_test_stack.shape[0], X_test_stack.shape[1], X_test_stack.shape[2]])  
    
    
    print('X_train_stack:',X_train_stack.shape)
    print('X_test_stack:',X_test_stack.shape)
    print('y_Acc_train_stack',y_Acc_train_stack.shape)
    print('y_Acc_test_stack',y_Acc_test_stack.shape)
    print('y_Disp_train_stack',y_Disp_train_stack.shape)
    print('y_Disp_test_stack',y_Disp_test_stack.shape)
    print('y_StoryDrift_train_stack',y_StoryDrift_train_stack.shape)
    print('y_StoryDrift_test_stack',y_StoryDrift_test_stack.shape)
    
    
    scipy.io.savemat( level[0]+'/NewData_Acc.mat',
                     {'X_train_stack':X_train_stack,'X_test_stack':X_test_stack,'y_Acc_train_stack':y_Acc_train_stack,'y_Acc_test_stack':y_Acc_test_stack,
                      'name_train':name_train,'name_test':name_test,'step_train':step_train,'step_test':step_test,
                      'trainInd1':trainInd1,'testInd1':testInd1,'trainInd2':trainInd2,'testInd2':testInd2})
    
    scipy.io.savemat( level[0]+'/NewData_Disp.mat',
                     {'X_train_stack':X_train_stack,'X_test_stack':X_test_stack,'y_Disp_train_stack':y_Disp_train_stack,'y_Disp_test_stack':y_Disp_test_stack,
                      'name_train':name_train,'name_test':name_test,'step_train':step_train,'step_test':step_test,
                      'trainInd1':trainInd1,'testInd1':testInd1,'trainInd2':trainInd2,'testInd2':testInd2})
    
    scipy.io.savemat( level[0]+'/NewData_StoryDrift.mat',
                     {'X_train_stack':X_train_stack,'X_test_stack':X_test_stack,'y_StoryDrift_train_stack':y_StoryDrift_train_stack,'y_StoryDrift_test_stack':y_StoryDrift_test_stack,
                      'name_train':name_train,'name_test':name_test,'step_train':step_train,'step_test':step_test,
                      'trainInd1':trainInd1,'testInd1':testInd1,'trainInd2':trainInd2,'testInd2':testInd2})
 
 



























