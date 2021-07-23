
'''

The results obtained using and modifying this code and hydration data set may be used in any publications provided that its use is explicitly acknowledged. 
A suitable reference is: Junji Hyodo, Kota Tsujikawa, Motoki Shiga, Yuji Okuyama, and Yoshihiro Yamazaki*, “Accelerated discovery of proton-conducting perovskite oxide by capturing physicochemical fundamentals of hydration”, ACS Energy Letters, 6(2021) xxx-xxx.  (DOI:xxxx)

'''



import os
import shutil
import itertools
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# our developed library
from mylib import *

if __name__=='__main__':

    # generate output directry
    dir_root_output = os.path.join('output','prediction')
    if os.path.exists(dir_root_output):
        shutil.rmtree(dir_root_output)
    os.mkdir(dir_root_output)

    # generate experiment condition directry
    dir_vec_output = os.path.join(dir_root_output,'VirtualExpCondition')
    os.mkdir(dir_vec_output)
    
    # generate experiment dataset directry
    dir_ve_output = os.path.join(dir_root_output,'VirtualExperiment')
    os.mkdir(dir_ve_output)

    # generate CH prediction directry
    dir_CH_pred_output = os.path.join(dir_root_output,'predict_CH')
    os.mkdir(dir_CH_pred_output)

    #---------------------------------------------------------------------------

    # predict CH
    # prefix file name of a virtial experimental dataset
    file_name_base = 'experimental_conditions_test_ABB1O3'
    # select experiment condiciton
    list_temperature = np.arange(400, 1001, 50)
    pH2O = 0.02 
    SinteringTemperature = 1600
    SinteringTime = 24

    # generate file of experiment dataset of virtual composition
    generate_experimental_condition(dir_vec_output, file_name_base,
                    list_temperature, pH2O,SinteringTemperature,SinteringTime)

    # select data file of elemental information
    file_element = os.path.join('data','elements_data.csv')
    # file name of experiment condition data for test data
    file_test = os.path.join(dir_vec_output,file_name_base+'_all.csv')
    # file name of predictin data
    file_virtual_experiment_test = os.path.join(dir_ve_output,'virtual_experiment_dataset_test.csv')
    
    # generate descriptors from experimental conditions and chemical compositions
    combine_dataset(file_element, file_test, file_virtual_experiment_test, None)

    # threshold of CH/Cdopant
    threshold_CHCdopant = 10**(-2) 
    # select list of descriptor
    file_feature = os.path.join('data','features.csv')
    # select machine learning model
    model = GradientBoostingRegressor(n_estimators=500,random_state=0)
    # standard scaler
    flag_StandardScaler = False
    # Variables to predict
    flag_proc = 'log_CHdopant' 

    # file name of experimental data
    file_train = os.path.join('data','Data S1.csv')
    # output file name of training data, which is combinations of experimental data and descriptors
    file_virtual_experiment_train = os.path.join(dir_ve_output,'virtual_experiment_dataset_train.csv')
    # generate train dataset
    combine_dataset(file_element, file_train, file_virtual_experiment_train, threshold_CHCdopant)
    # train a machine learning model and predict CH
    virtual_prediction(file_virtual_experiment_train, file_virtual_experiment_test,
            file_feature, dir_CH_pred_output, model, flag_StandardScaler, flag_proc)


