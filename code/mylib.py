
'''

The results obtained using and modifying this code and hydration data set may be used in any publications provided that its use is explicitly acknowledged. 
A suitable reference is: Junji Hyodo, Kota Tsujikawa, Motoki Shiga, Yuji Okuyama, and Yoshihiro Yamazaki*, “Accelerated discovery of proton-conducting perovskite oxide by capturing physicochemical fundamentals of hydration”, ACS Energy Letters, 6(2021) xxx-xxx.  (DOI:xxxx)

'''


import os
import shutil
import itertools
import re
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

def generate_experimental_condition(dir_output, file_name_prefix,list_temperature, partial_pressure_H2O=0.02,SinteringTemperature=1600,SinteringTime=24):
    """
    Generate experimental condition data to a csv file

    Parameters
    ----------
    dir_output : str
        output directory
    file_name_prefix : str
        name prefix of output file
    list_temperature : list of int
        list of experimental temperature (Celsius)
    partial_pressure_H2O : float
        experimental water vapor partial pressure
    SinteringTemperature : int
        sintering temperature
    SinteringTime : int
        sintering time

    Returns
    -------
    None

    """

    print("Enter the host element occupying the A-site")
    set_A1 = input ("Ex: Ba\n")
    print("Enter the valence of the A-site host element")
    set_A1_valence = input("Ex: 2\n")
    frac_A1 = '1'
    print("Enter the host element occupying the B-site")
    set_B1 = input ("Ex: Zr\n")
    print("Enter the valence of the B-site host element")
    set_B1_valence = input("Ex:4\n")
    print("Enter the fraction that describes the composition of the B-site host element")
    frac_B1 = str(format(float( input ("Ex:0.8\n")), '.2f'))
    print("Enter the dopant element occupying the B-site")
    set_B2 = input ("Ex: Sc\n")
    print("Enter the valence of the B-dopant")
    set_B2_valence = input("Ex: 3\n")
    frac_B2 = str(format((1 - float(frac_B1)), '.2f'))

    # generate dataframe for base
    CA = set_A1 + set_B1 + frac_B1 + set_B2 + frac_B2 + "O3"
    dic = {'Composition':CA,
           'A1':set_A1, 'Valence A1':set_A1_valence, 'fraction A1':frac_A1,
           'B1':set_B1, 'Valence B1':set_B1_valence, 'fraction B1':frac_B1,
           'B2':set_B2, 'Valence B2':set_B2_valence, 'fraction B2':frac_B2}
    df = pd.DataFrame(dic,index=['i',])

    # add columns name
    columns_all = ['Composition','Temperature / C','pH2O / atm','CH',
                'A1','Valence A1','fraction A1','A2','Valence A2','fraction A2',
                'B1','Valence B1','fraction B1','B2','Valence B2','fraction B2',
                'B3','Valence B3','fraction B3','X1','Valence X1','fraction X1','fraction total']
    for c in columns_all:
        if not(c in df.columns):
            df[c] = float(np.NaN)
    df = df[columns_all]

    # add another experimental conditions
    df['pH2O / atm'] = partial_pressure_H2O
    df['Sintering temperature/C'] = SinteringTemperature
    df['Sintering time / h'] = SinteringTime
    df['fraction A2']='0'
    df['fraction B3']='0'
    df['X1']='O'
    df['Valence X1']='-2'
    df['fraction X1']='0.2'
    df['fraction total']='1'

    for cnt, tmp in enumerate(list_temperature):
        df['Temperature / C'] = tmp
        if cnt==0:
            df_all = df.copy()
        else:
            df_all = pd.concat([df_all,df], ignore_index=True)
    file_name = os.path.join(dir_output,'{:}_all.csv'.format(file_name_prefix, tmp))
    df_all.to_csv(file_name, index=False)


#------------------------------------------------------------
def predict_CH(X_train, y_train, dopant_fraction_train, X_test, dopant_fraction_test, model, flag_StandardScaler, flag_proc):
    """
    train and predict proton concentration by machine learning model

    Parameters
    ----------
    X_train : numpy.array
        dataset of training data
    y_train : numpy.array
        target variable of training data
    dopant_fraction_train :  : numpy.array
        dopant fractions of training data
    X_test : numpy.array
        dataset of test data
    dopant_fraction_train :  : numpy.array
        dopant fractions of test data
    model : scikit-learn model
        machine learning model
    flag_StandardScaler : Boolean
        with or without standardization (True or False)
    flag_proc : str
        target variable used for a machine learning model
        (convert to the proton concentration of the objective variable when prediction）
        choose from the following 4 options:
          'direct' : proton concentration
          'log' : logarithm of proton concentration
          'CHdopant' : ratio of proton concentration and dopant fraction
          'log_CHdopant' : logarithm of ratio of proton concentration and dopant fraction

    Returns
    -------
    y_predict　: numpy.array
        predicted value (proton concentration) of test data
    """

    # set random seed
    if 'random_state' in model.get_params().keys():
        model = model.set_params(random_state=0)

    # standardize each descriptor
    if flag_StandardScaler:
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

    # training and prediction
    if flag_proc == 'log_CHdopant':
        model.fit(X_train, np.log(y_train/dopant_fraction_train))
        y_pred = np.exp(model.predict(X_test))
        y_pred[y_pred>1] = 1
        y_pred = y_pred*dopant_fraction_test
        y_pred[y_pred < 0] = 0
    elif flag_proc == 'direct':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred[y_pred>1] = 1
        y_pred[y_pred < 0] = 0
    elif flag_proc == 'CHdopant':
        model.fit(X_train, y_train/dopant_fraction_train)
        y_pred = model.predict(X_test)
        y_pred[y_pred>1] = 1
        y_pred = y_pred*dopant_fraction_test
        y_pred[y_pred<0]=0
    elif flag_proc == 'log':
        model.fit(X_train, np.log(y_train))
        y_pred = np.exp(model.predict(X_test))
        y_pred[y_pred>1] = 1
        y_pred[y_pred < 0] = 0

    # output prediction for test data
    return y_pred


#------------------------------------------------------------
def feature_ranking(X_train, y_train, dopant_fraction_train, list_feature, model, flag_proc):
    """
    descriptors ranking by importance scores of machine learning model

    Parameters
    ----------
    X_train : numpy.array
        dataset of training dataset
    y_train : numpy.array
        target variable of training dataset
    dopant_fraction_train :  : numpy.array
        dopant fractions of training data
    list_feature : list
        list of descriptor names
    model : scikit-learn model
        machine learning model
    flag_proc : str
        target variable used for a machine learning model
        (convert to the proton concentration of the objective variable when prediction）
        choose from the following 4 options:
          'direct' : proton concentration
          'log' : logarithm of proton concentration
          'CHdopant' : ratio of proton concentration and dopant fraction
          'log_CHdopant' : logarithm of ratio of proton concentration and dopant fraction

    Returns
    -------
    df_ranking　: pandas.DataFrame
        ranked list of descriptor names with importance scores


    """
    # set random seed
    if 'random_state' in model.get_params().keys():
        model = model.set_params(random_state=0)

    # tarining model for descriptor ranking
    if flag_proc == 'log_CHdopant':
        model.fit(X_train, np.log(y_train/dopant_fraction_train))
    elif flag_proc == 'direct':
        model.fit(X_train, y_train)
    elif flag_proc == 'CHdopant':
        model.fit(X_train, y_train/dopant_fraction_train)
    elif flag_proc == 'log':
        model.fit(X_train, np.log(y_train))
    fi = model.feature_importances_

    # sort descriptors by importance scores
    i = np.argsort(-fi)
    df_ranking = pd.DataFrame({'Name':list_feature[i], 'Importance':fi[i]})
    df_ranking = df_ranking[['Name','Importance']]

    # output DataFrame of ranked descriptors and importance scores
    return df_ranking


#------------------------------------------------------------
def virtual_prediction(file_train, file_test, file_feature, dir_output, model, flag_StandardScaler, flag_proc):
    """
    Predicte proton concentration of virtual composition

    Parameters
    ----------
    file_train : str
        file name of training data
    file_test : str
        file name of test data
    file_feature : str
        file name of descriptors
    dir_output : str
        output directory
    model : model of scikit-learn
        machine learning model to use
    flag_StandardScaler : boolean
        with or without standardization (True or False)
    flag_proc : str
        target variable used for a machine learning model
        (convert to the proton concentration of the objective variable when prediction）
        choose from the following 4 options:
          'direct' : proton concentration
          'log' : logarithm of proton concentration
          'CHdopant' : ratio of proton concentration and dopant fraction
          'log_CHdopant' : logarithm of ratio of proton concentration and dopant fraction

    Returns
    -------
    None
    """

    # load dataset of train and test dataset
    df_train = pd.read_csv(file_train,index_col=False)
    df_test = pd.read_csv(file_test,index_col=False)

    # load list of descriptors
    fea_list = np.array(pd.read_csv(file_feature,index_col=False).columns)

    # convert datasets in Pandas dataframes to numpy arrays
    X_train = np.array(df_train[fea_list])
    y_train = np.array(df_train['CH'])
    X_test = np.array(df_test[fea_list])
    dopant_fraction_train = np.array(df_train['dopant fraction'])
    dopant_fraction_test = np.array(df_test['dopant fraction'])

    # predict CH of test data
    y_pred = predict_CH(X_train, y_train, dopant_fraction_train, X_test,
                dopant_fraction_test, model, flag_StandardScaler, flag_proc)

    # extraction of experimental conditions of test data
    chemav = df_test['Composition']
    tempe = df_test['Temperature / C'] 
    p_H2O = df_test['pH2O / atm']

    # output prediction results to a csv file
    df_out = pd.DataFrame({'Composition':chemav,'Temperature / C':tempe, 'CH_predicted':y_pred, 'Dopant fraction':dopant_fraction_test, 'p_H2O / atm':p_H2O})
    df_out = df_out.sort_values(['Composition','Temperature / C'])
    df_out.to_csv(os.path.join(dir_output,'prediction_all.csv'),index=False)

    # output importance scores of descriptors to a csv file
    if hasattr(model,'feature_importances_'):
        df_ranking = feature_ranking(X_train, y_train, dopant_fraction_train, fea_list, model, flag_proc)
        df_ranking.to_csv(os.path.join(dir_output,'fea_importance_all.csv'),index=False)


#------------------------------------------------------------
def combine_dataset(file_element, file_experiment, file_save, threshold=None):
    """
    generate dataset for machine learning by combining elemental information and experimental conditions.

    Parameters
    ----------
    file_element : str
        file name of elemental imformation
    file_experiment : str
        file name of experimental conditions
    file_save : str
        fime name of output data
    threshold : float
        threshold of ratio of proton concentration and dopant fraction
        to exclude experimental data below the threshold.
        If threshold is not specified, any data is not excluded

    Returns
    -------
    None

    """

    # load elemental data file
    df_fea = pd.read_csv(file_element)

    # load experimental conditions
    tmp, ext = os.path.splitext(file_experiment)
    if ext=='.xlsx':
        df_exp = pd.read_csv(file_experiment)
    elif ext=='.csv':
        df_exp = pd.read_csv(file_experiment)
    N = df_exp.shape[0]

    # caclulate A site fraction
    Const_A       = np.array(df_exp['fraction A1'] + df_exp['fraction A2'])
    fraction_A1 = np.array(df_exp['fraction A1'] / Const_A)
    fraction_A2 = np.array(df_exp['fraction A2'] / Const_A)

    # caclulate B site fraction
    Const_B       = np.array(df_exp['fraction B1'] + df_exp['fraction B2'] + df_exp['fraction B3'])
    fraction_B1 = np.array(df_exp['fraction B1'] / Const_B)
    fraction_B2 = np.array(df_exp['fraction B2'] / Const_B)
    fraction_B3 = np.array(df_exp['fraction B3'] / Const_B)

    # descriptors of oxygen
    fraction_X1 = np.ones(N)*3
    aw_X1 = np.ones(N)*16
    ir_X1 = np.ones(N)*1.4

    # determine if A2, B2, B3 are dopants
    flag_dp_A2 = np.array(df_exp['Valence A2'] < df_exp['Valence A1'])*1
    flag_dp_B2 = np.array(df_exp['Valence B2'] < df_exp['Valence B1'])*1
    flag_dp_B3 = np.array(df_exp['Valence B3'] < df_exp['Valence B1'])*1

    # determine if A2, B2, B3 are hosts
    flag_ht_A2 = 1-flag_dp_A2
    flag_ht_B2 = 1-flag_dp_B2
    flag_ht_B3 = 1-flag_dp_B3

    # ratio host and dopant 
    dopant_fraction = flag_dp_A2*fraction_A2 + flag_dp_B2*fraction_B2 + flag_dp_B3*fraction_B3
    host_fraction = 2 - dopant_fraction
    dopant_A_fraction =  flag_dp_A2*fraction_A2
    host_A_fraction =  1 - dopant_A_fraction
    dopant_B_fraction =  flag_dp_B2*fraction_B2 + flag_dp_B3*fraction_B3
    host_B_fraction = 1 - dopant_B_fraction

    # target variable (proton concentration)
    CH = np.array(df_exp['CH'])

    # extract elemental information of A1 site
    aw_A1, ad_A1, mp_A1, fie_A1, en_A1, ir_A1 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for n in range(N):
        if fraction_A1[n]>0:
            a = df_exp['A1'][n]
            v = df_exp['Valence A1'][n]
            i = np.where((df_fea['Atom']==a)&(df_fea['Valence']==v))[0]
            if len(i)==0:
                print('None!')
            aw_A1[n] = df_fea['Atomic weight'][i]
            ad_A1[n] = df_fea['Atomic density'][i]
            mp_A1[n] = df_fea['Melting point'][i]
            fie_A1[n] = df_fea['First ionization energy'][i]
            en_A1[n] = df_fea['Electronegativity'][i]
            ir_A1[n] = df_fea['Ionic radius XII'][i]

    # extract elemental information of A2 site
    aw_A2, ad_A2, mp_A2, fie_A2, en_A2, ir_A2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for n in range(N):
        if fraction_A2[n]>0:
            a = df_exp['A2'][n]
            v = df_exp['Valence A2'][n]
            i = np.where((df_fea['Atom']==a)&(df_fea['Valence']==v))[0]
            if len(i)==0:
                print('None!')
            aw_A2[n] = df_fea['Atomic weight'][i]
            ad_A2[n] = df_fea['Atomic density'][i]
            mp_A2[n] = df_fea['Melting point'][i]
            fie_A2[n] = df_fea['First ionization energy'][i]
            en_A2[n] = df_fea['Electronegativity'][i]
            ir_A2[n] = df_fea['Ionic radius XII'][i]

    # extract elemental information of B1 site
    aw_B1, ad_B1, mp_B1, fie_B1, en_B1, ir_B1 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for n in range(N):
        if fraction_B1[n]>0:
            a = df_exp['B1'][n]
            v = df_exp['Valence B1'][n]
            i = np.where((df_fea['Atom']==a)&(df_fea['Valence']==v))[0]
            if len(i)==0:
                print('None!')
            aw_B1[n] = df_fea['Atomic weight'][i]
            ad_B1[n] = df_fea['Atomic density'][i]
            mp_B1[n] = df_fea['Melting point'][i]
            fie_B1[n] = df_fea['First ionization energy'][i]
            en_B1[n] = df_fea['Electronegativity'][i]
            ir_B1[n] = df_fea['Ionic radius VI'][i]

    # extract elemental information of B2 site
    aw_B2, ad_B2, mp_B2, fie_B2, en_B2, ir_B2 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for n in range(N):
        if fraction_B2[n]>0:
            a = df_exp['B2'][n]
            v = df_exp['Valence B2'][n]
            i = np.where((df_fea['Atom']==a)&(df_fea['Valence']==v))[0]
            if len(i)==0:
                print('None!')
            aw_B2[n] = df_fea['Atomic weight'][i]
            ad_B2[n] = df_fea['Atomic density'][i]
            mp_B2[n] = df_fea['Melting point'][i]
            fie_B2[n] = df_fea['First ionization energy'][i]
            en_B2[n] = df_fea['Electronegativity'][i]
            ir_B2[n] = df_fea['Ionic radius VI'][i]

    # extract elemental information of B3 site
    aw_B3, ad_B3, mp_B3, fie_B3, en_B3, ir_B3 = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for n in range(N):
        if fraction_B3[n]>0:
            a = df_exp['B3'][n]
            v = df_exp['Valence B3'][n]
            i = np.where((df_fea['Atom']==a)&(df_fea['Valence']==v))[0]
            if len(i)==0:
                print('None!')
            aw_B3[n] = df_fea['Atomic weight'][i]
            ad_B3[n] = df_fea['Atomic density'][i]
            mp_B3[n] = df_fea['Melting point'][i]
            fie_B3[n] = df_fea['First ionization energy'][i]
            en_B3[n] = df_fea['Electronegativity'][i]
            ir_B3[n] = df_fea['Ionic radius VI'][i]

    #----------------------------------------------------------------------------
    # calculate descriptors
    ave_aw_A = fraction_A1*aw_A1 + fraction_A2*aw_A2
    ave_ad_A = fraction_A1*ad_A1 + fraction_A2*ad_A2
    ave_mp_A = fraction_A1*mp_A1 + fraction_A2*mp_A2
    ave_fie_A = fraction_A1*fie_A1 + fraction_A2*fie_A2
    ave_en_A = fraction_A1*en_A1 + fraction_A2*en_A2
    ave_ir_A = fraction_A1*ir_A1 + fraction_A2*ir_A2

    ave_aw_B = fraction_B1*aw_B1 + fraction_B2*aw_B2 + fraction_B3*aw_B3
    ave_ad_B = fraction_B1*ad_B1 + fraction_B2*ad_B2 + fraction_B3*ad_B3
    ave_mp_B = fraction_B1*mp_B1 + fraction_B2*mp_B2 + fraction_B3*mp_B3
    ave_fie_B = fraction_B1*fie_B1 + fraction_B2*fie_B2 + fraction_B3*fie_B3
    ave_en_B = fraction_B1*en_B1 + fraction_B2*en_B2 + fraction_B3*en_B3
    ave_ir_B = fraction_B1*ir_B1 + fraction_B2*ir_B2 + fraction_B3*ir_B3

    Molar_weight = ave_aw_A + ave_aw_B + fraction_X1*aw_X1
    T_sinter_time_K_h = np.array(df_exp["Sintering temperature/C"] * df_exp["Sintering time / h"])

    ratio_aw_AB = ave_aw_A / ave_aw_B
    ratio_ad_AB = ave_ad_A / ave_ad_B
    ratio_mp_AB = ave_mp_A / ave_mp_B
    ratio_fie_AB = ave_fie_A / ave_fie_B
    ratio_en_AB = ave_en_A / ave_en_B
    ratio_ir_AB = ave_ir_A / ave_ir_B

    ave_aw_host =  ( fraction_A1*aw_A1 + fraction_B1*aw_B1
                    + flag_ht_A2*fraction_A2*aw_A2 + flag_ht_B2*fraction_B2*aw_B2 + flag_ht_B3*fraction_B3*aw_B3  ) / host_fraction
    ave_ad_host =  ( fraction_A1*ad_A1 + fraction_B1*ad_B1
                   + flag_ht_A2*fraction_A2*ad_A2 + flag_ht_B2*fraction_B2*ad_B2 + flag_ht_B3*fraction_B3*ad_B3) / host_fraction
    ave_mp_host =  ( fraction_A1*mp_A1 + fraction_B1*mp_B1
                   + flag_ht_A2*fraction_A2*mp_A2 + flag_ht_B2*fraction_B2*mp_B2 + flag_ht_B3*fraction_B3*mp_B3) / host_fraction
    ave_fie_host =  ( fraction_A1*fie_A1 + fraction_B1*fie_B1
                    + flag_ht_A2*fraction_A2*fie_A2 + flag_ht_B2*fraction_B2*fie_B2 + flag_ht_B3*fraction_B3*fie_B3) / host_fraction
    ave_en_host =  ( fraction_A1*en_A1 + fraction_B1*en_B1
                   + flag_ht_A2*fraction_A2*en_A2 + flag_ht_B2*fraction_B2*en_B2 + flag_ht_B3*fraction_B3*en_B3) / host_fraction
    ave_ir_host =  ( fraction_A1*ir_A1 + fraction_B1*ir_B1
                   + flag_ht_A2*fraction_A2*ir_A2 + flag_ht_B2*fraction_B2*ir_B2 + flag_ht_B3*fraction_B3*ir_B3) / host_fraction

    ave_aw_dopant =  ( flag_dp_A2*fraction_A2*aw_A2 + flag_dp_B2*fraction_B2*aw_B2 + flag_dp_B3*fraction_B3*aw_B3  ) / dopant_fraction
    ave_ad_dopant =  ( flag_dp_A2*fraction_A2*ad_A2 + flag_dp_B2*fraction_B2*ad_B2 + flag_dp_B3*fraction_B3*ad_B3) / dopant_fraction
    ave_mp_dopant =  ( flag_dp_A2*fraction_A2*mp_A2 + flag_dp_B2*fraction_B2*mp_B2 + flag_dp_B3*fraction_B3*mp_B3) / dopant_fraction
    ave_fie_dopant =  ( flag_dp_A2*fraction_A2*fie_A2 + flag_dp_B2*fraction_B2*fie_B2 + flag_dp_B3*fraction_B3*fie_B3) / dopant_fraction
    ave_en_dopant =  ( flag_dp_A2*fraction_A2*en_A2 + flag_dp_B2*fraction_B2*en_B2 + flag_dp_B3*fraction_B3*en_B3) / dopant_fraction
    ave_ir_dopant =  (  flag_dp_A2*fraction_A2*ir_A2 + flag_dp_B2*fraction_B2*ir_B2 + flag_dp_B3*fraction_B3*ir_B3) / dopant_fraction

    ave_aw_host_A =  ( fraction_A1*aw_A1 + flag_ht_A2*fraction_A2*aw_A2  ) / host_A_fraction
    ave_ad_host_A =  ( fraction_A1*ad_A1+ flag_ht_A2*fraction_A2*ad_A2 ) / host_A_fraction
    ave_mp_host_A =  ( fraction_A1*mp_A1+ flag_ht_A2*fraction_A2*mp_A2) / host_A_fraction
    ave_fie_host_A =  ( fraction_A1*fie_A1 + flag_ht_A2*fraction_A2*fie_A2) / host_A_fraction
    ave_en_host_A =  ( fraction_A1*en_A1+ flag_ht_A2*fraction_A2*en_A2) / host_A_fraction
    ave_ir_host_A =  ( fraction_A1*ir_A1+ flag_ht_A2*fraction_A2*ir_A2) / host_A_fraction

    ave_aw_host_B =  ( fraction_B1*aw_B1 + flag_ht_B2*fraction_B2*aw_B2 + flag_ht_B3*fraction_B3*aw_B3  ) / host_B_fraction
    ave_ad_host_B =  ( fraction_B1*ad_B1+ flag_ht_B2*fraction_B2*ad_B2 + flag_ht_B3*fraction_B3*ad_B3) / host_B_fraction
    ave_mp_host_B =  ( fraction_B1*mp_B1 + flag_ht_B2*fraction_B2*mp_B2 + flag_ht_B3*fraction_B3*mp_B3) / host_B_fraction
    ave_fie_host_B =  ( fraction_B1*fie_B1 + flag_ht_B2*fraction_B2*fie_B2 + flag_ht_B3*fraction_B3*fie_B3) / host_B_fraction
    ave_en_host_B =  ( fraction_B1*en_B1 + flag_ht_B2*fraction_B2*en_B2 + flag_ht_B3*fraction_B3*en_B3) / host_B_fraction
    ave_ir_host_B =  ( fraction_B1*ir_B1 + flag_ht_B2*fraction_B2*ir_B2 + flag_ht_B3*fraction_B3*ir_B3) / host_B_fraction

    ratio_aw_host_B_host_A =  ave_aw_host_B / ave_aw_host_A
    ratio_ad_host_B_host_A =    ave_ad_host_B / ave_ad_host_A
    ratio_mp_host_B_host_A =    ave_mp_host_B / ave_mp_host_A
    ratio_fie_host_B_host_A =    ave_fie_host_B / ave_fie_host_A
    ratio_en_host_B_host_A =    ave_en_host_B / ave_en_host_A
    ratio_ir_host_B_host_A =   ave_ir_host_B / ave_ir_host_A

    ratio_aw_dopant_host_A =  ave_aw_dopant / ave_aw_host_A
    ratio_ad_dopant_host_A =    ave_ad_dopant / ave_ad_host_A
    ratio_mp_dopant_host_A =    ave_mp_dopant / ave_mp_host_A
    ratio_fie_dopant_host_A =    ave_fie_dopant / ave_fie_host_A
    ratio_en_dopant_host_A =    ave_en_dopant / ave_en_host_A
    ratio_ir_dopant_host_A =   ave_ir_dopant / ave_ir_host_A

    ratio_aw_dopant_host_B =  ave_aw_dopant / ave_aw_host_B
    ratio_ad_dopant_host_B =    ave_ad_dopant / ave_ad_host_B
    ratio_mp_dopant_host_B =    ave_mp_dopant / ave_mp_host_B
    ratio_fie_dopant_host_B =    ave_fie_dopant / ave_fie_host_B
    ratio_en_dopant_host_B =    ave_en_dopant / ave_en_host_B
    ratio_ir_dopant_host_B =   ave_ir_dopant / ave_ir_host_B

    ratio_aw_dopant_host =  ave_aw_dopant / ave_aw_host
    ratio_ad_dopant_host =    ave_ad_dopant / ave_ad_host
    ratio_mp_dopant_host =    ave_mp_dopant / ave_mp_host
    ratio_fie_dopant_host =    ave_fie_dopant / ave_fie_host
    ratio_en_dopant_host =    ave_en_dopant / ave_en_host
    ratio_ir_dopant_host =   ave_ir_dopant / ave_ir_host

    MW_ir_AB = Molar_weight / np.sqrt(ave_ir_A*ave_ir_B)
    QToleranceFactor = (ave_ir_A + ir_X1)/np.sqrt(2)/(ave_ir_B +  ir_X1)
    #----------------------------------------------------------------------------

    # combine information ( descriptors, experimental condition ) for output
    df_new_fea = {
        "CH":CH,
        "dopant fraction":dopant_fraction,
        "host fraction":host_fraction,
        "host_A fraction":host_A_fraction,
        "host_B fraction":host_B_fraction,
        "dopant_A fraction":dopant_A_fraction,
        "dopant_B fraction":dopant_B_fraction,
        "Molar_weight/gmol-1":Molar_weight,
        "ToleranceFactor":QToleranceFactor,
        "T_sinter*time / K h":T_sinter_time_K_h,
        "average atomic_weight_A":ave_aw_A,
        "average atomic_density_A":ave_ad_A,
        "average melting_point_A":ave_mp_A,
        "average first_ionization_energy_A":ave_fie_A,
        "average electronegativity_A":ave_en_A,
        "average ionic_radius_A":ave_ir_A,
        "average atomic_weight_B":ave_aw_B,
        "average atomic_density_B":ave_ad_B,
        "average melting_point_B":ave_mp_B,
        "average first_ionization_energy_B":ave_fie_B,
        "average electronegativity_B":ave_en_B,
        "average ionic_radius_B":ave_ir_B,
        "ratio atomic_weight_A/B":ratio_aw_AB,
        "ratio atomic_density_A/B": ratio_ad_AB,
        "ratio melting_point_A/B":ratio_mp_AB,
        "ratio first_ionization_energy_A/B":ratio_fie_AB,
        "ratio electronegativity_A/B":ratio_en_AB,
        "ratio ionic_radius_A/B":ratio_ir_AB,
        "average atomic_weight_host":ave_aw_host,
        "average atomic_density_host":ave_ad_host,
        "average melting_point_host":ave_mp_host,
        "average first_ionization_energy_host":ave_fie_host,
        "average electronegativity_host":ave_en_host,
        "average ionic_radius_host":ave_ir_host,
        "average atomic_weight_dopant":ave_aw_dopant,
        "average atomic_density_dopant":ave_ad_dopant,
        "average melting_point_dopant":ave_mp_dopant,
        "average first_ionization_energy_dopant":ave_fie_dopant,
        "average electronegativity_dopant":ave_en_dopant,
        "average ionic_radius_dopant":ave_ir_dopant,
        "average atomic_weight_host_A":ave_aw_host_A,
        "average atomic_density_host_A":ave_ad_host_A,
        "average melting_point_host_A":ave_mp_host_A,
        "average first_ionization_energy_host_A":ave_fie_host_A,
        "average electronegativity_host_A":ave_en_host_A,
        "average ionic_radius_host_A":ave_ir_host_A,
        "average atomic_weight_host_B":ave_aw_host_B,
        "average atomic_density_host_B":ave_ad_host_B,
        "average melting_point_host_B":ave_mp_host_B,
        "average first_ionization_energy_host_B":ave_fie_host_B,
        "average electronegativity_host_B":ave_en_host_B,
        "average ionic_radius_host_B":ave_ir_host_B,
        "ratio atomic_weight_host_A/B":ratio_aw_host_B_host_A,
        "ratio atomic_density_host_A/B":ratio_ad_host_B_host_A,
        "ratio melting_point_host_A/B":ratio_mp_host_B_host_A,
        "ratio first_ionization_energy_host_A/B":ratio_fie_host_B_host_A,
        "ratio electronegativity_host_A/B":ratio_en_host_B_host_A,
        "ratio ionic_radius_host_A/B":ratio_ir_host_B_host_A,
        "ratio atomic_weight_dopant/host_A":ratio_aw_dopant_host_A,
        "ratio atomic_density_dopant/host_A":ratio_ad_dopant_host_A,
        "ratio melting_point_dopant/host_A":ratio_mp_dopant_host_A,
        "ratio first_ionization_energy_dopant/host_A":ratio_fie_dopant_host_A,
        "ratio electronegativity_dopant/host_A":ratio_en_dopant_host_A,
        "ratio ionic_radius_dopant/host_A":ratio_ir_dopant_host_A,
        "ratio atomic_weight_dopant/host_B":ratio_aw_dopant_host_B,
        "ratio atomic_density_dopant/host_B":ratio_ad_dopant_host_B,
        "ratio melting_point_dopant/host_B":ratio_mp_dopant_host_B,
        "ratio first_ionization_energy_dopant/host_B":ratio_fie_dopant_host_B,
        "ratio electronegativity_dopant/host_B":ratio_en_dopant_host_B,
        "ratio ionic_radius_dopant/host_B":ratio_ir_dopant_host_B,
        "ratio atomic_weight_dopant/host":ratio_aw_dopant_host,
        "ratio atomic_density_dopant/host":ratio_ad_dopant_host,
        "ratio melting_point_dopant/host":ratio_mp_dopant_host,
        "ratio first_ionization_energy_dopant/host":ratio_fie_dopant_host,
        "ratio electronegativity_dopant/host":ratio_en_dopant_host,
        "ratio ionic_radius_dopant/host":ratio_ir_dopant_host,
        "Mw/sqrt(radiiA*radiiB)":MW_ir_AB
    }
    df_new_fea = pd.DataFrame(df_new_fea)

    # combine DataFrame of experimental conditions and descriptors
    fea_names = ['Composition', 'Temperature / C',
                 'pH2O / atm', 'Sintering temperature/C', 'Sintering time / h',
                 'fraction A1', 'fraction A2', 'fraction B1', 'fraction B2', 'fraction B3']
    df_new = pd.concat([df_exp[fea_names], df_new_fea], axis=1)

    # rearange descriptors and target variable (CH)
    n = len(fea_names)
    col = [df_new.columns[0]] + [df_new.columns[n]] + list(df_new.columns[1:n]) + list(df_new.columns[(n+1):])
    df_new = df_new[col]

    # exclude data that the ratio of proton concentration and dopant below the threshold
    if threshold:
        i = np.where(df_new['CH']/df_new['dopant fraction']>=threshold)[0]
        df_new = df_new.iloc[i,:]

    # save of dataset to a csv file
    df_new = df_new.sort_values(['Composition','Temperature / C'])
    df_new.to_csv(file_save, index=False, float_format='%.3f')

