import gc

import os
import json
import logging
import sys
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # to apply PCA



def df_all_creator(data_filepath, sampling):
    """

     """
    # Time tracking, Operation time (min):  0.003
    t = time.process_time()

    with h5py.File(data_filepath, 'r') as hdf:
        # Development(training) set
        W_dev = np.array(hdf.get('W_dev'))  # W
        X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
        X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
        T_dev = np.array(hdf.get('T_dev'))  # T
        Y_dev = np.array(hdf.get('Y_dev'))  # RUL
        A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))  # W
        X_s_test = np.array(hdf.get('X_s_test'))  # X_s
        X_v_test = np.array(hdf.get('X_v_test'))  # X_v
        T_test = np.array(hdf.get('T_test'))  # T
        Y_test = np.array(hdf.get('Y_test'))  # RUL
        A_test = np.array(hdf.get('A_test'))  # Auxiliary

        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        X_v_var = np.array(hdf.get('X_v_var'))
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))
        X_v_var = list(np.array(X_v_var, dtype='U20'))
        T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))


    W = np.concatenate((W_dev, W_test), axis=0)
    X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
    X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
    T = np.concatenate((T_dev, T_test), axis=0)
    Y = np.concatenate((Y_dev, Y_test), axis=0)
    A = np.concatenate((A_dev, A_test), axis=0)

    print('')
    print("Operation time (min): ", (time.process_time() - t) / 60)
    print("number of training samples(timestamps): ", Y_dev.shape[0])
    print("number of test samples(timestamps): ", Y_test.shape[0])
    print('')
    print("W shape: " + str(W.shape))
    print("X_s shape: " + str(X_s.shape))
    print("X_v shape: " + str(X_v.shape))
    print("T shape: " + str(T.shape))
    print("Y shape: " + str(Y.shape))
    print("A shape: " + str(A.shape))

    '''
    Illusration of Multivariate time-series of condition monitoring sensors readings for Unit5 (fifth engine)

    W: operative conditions (Scenario descriptors) - ['alt', 'Mach', 'TRA', 'T2']
    X_s: measured signals - ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
    X_v: virtual sensors - ['T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi']
    T(theta): engine health parameters - ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod', 'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod', 'LPT_eff_mod', 'LPT_flow_mod']
    Y: RUL [in cycles]
    A: auxiliary data - ['unit', 'cycle', 'Fc', 'hs']
    '''

    df_W = pd.DataFrame(data=W, columns=W_var)
    df_Xs = pd.DataFrame(data=X_s, columns=X_s_var)
    df_Xv = pd.DataFrame(data=X_v[:,0:2], columns=['T40', 'P30'])
    df_T = pd.DataFrame(data=T, columns=T_var)
    df_Y = pd.DataFrame(data=Y, columns=['RUL'])
    df_A = pd.DataFrame(data=A, columns=A_var).drop(columns=['cycle', 'Fc', 'hs'])

    # Merge all the dataframes
    df_all = pd.concat([df_W, df_Xs, df_Xv,df_A,df_T, df_Y], axis=1)
    print ("df_all", df_all)    
    print ("df_all.shape", df_all.shape)
    df_all_smp = df_all[::sampling]
    print ("df_all_sub", df_all_smp)    
    print ("df_all_sub.shape", df_all_smp.shape)
    return df_all_smp

def df_train_creator(df_all, units_index_train):
    train_strata = []
    vlad_strata = []
    global pca, scaler
    #df_train_pca, Train_RUL, Train_Featuers, train_scaled, train_pca, df_train_scaled, _tempdata
    # global df_train, df_train_temp, df_train_concate, count_values, df_train_temp_unit_2, df_train_temp_2,df_train_1
    for idx in units_index_train:
        # selecting the training units
        df_train_temp = df_all[df_all['unit'] == np.float64(idx)]

        #Getting the number of datapoints in each RUL for unit 2
        count_values =  df_train_temp.value_counts('RUL', sort = False)

        # calculation for trainable samples for each RUL and unit

        count_values= list(count_values)
        #Sum of all samples in unit 2
        total_data_point_per_unit=sum(count_values)

       #Taking stratified random samples for each unit
        B=0
        sample_per_RUL=0
        for rul in count_values:
            #Getting required samples for each RUL
            sample_per_RUL_train=int(round(rul/total_data_point_per_unit*12050, 0 ))# # For (8,1) Train_per_unit ==  13506 to get 108046 (total)
            df_train_temp_1 = df_train_temp[df_train_temp['RUL']==np.float64(B)]
            #print(df_train_temp_1)
            #Sampling randomly for each required rows per RUL for training
            df_train_temp_2 = df_train_temp_1.sample(n=sample_per_RUL_train,axis =0)
            #Getting the resultant of the dataframe for testing
            df_vlad_test_temp_1 = df_train_temp_1.drop(df_train_temp_2.index)

            # here we are taking random samples from the returned Dataframe to have the exact amount required for testing
            sample_per_RUL_vlad=int(round(rul/total_data_point_per_unit*9050, 0 )) # For (8,1) Vald_per_unit ==  10231 to get 81450 (total)
            df_vlad_temp_2 = df_vlad_test_temp_1[df_vlad_test_temp_1['RUL']==np.float64(B)]
            #Sampling randomly for each required rows per RUL for testing
            df_vlad_temp_2 = df_vlad_temp_2.sample(n=sample_per_RUL_vlad,axis =0)

            B=B+1

            train_strata.append(df_train_temp_2)
            df_train_unit_strata = pd.concat(train_strata)
            #Sorting the training dataframe in ascending order with respect to unit number, RUL number and alt number
            df_train = df_train_unit_strata.sort_values(by = ["unit","RUL","alt"], ascending=[True, False, True])
           
        
            vlad_strata.append(df_vlad_temp_2)
            df_vlad=pd.concat(vlad_strata)
            #Sorting the testing
            # dataframe in ascending order with respect to unit number, RUL number and alt number
            df_vlad = df_vlad.sort_values(by = ["unit","RUL","alt"], ascending=[True, False, True])

            Train_Unit = df_train['unit'].reset_index(drop = True)
            Train_Featuers = df_train.iloc[:,0:36].reset_index(drop = True)
            Train_RUL = df_train['RUL'].reset_index(drop = True)

            
            # Applying fit_transform on Training DataSet             
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(Train_Featuers)
            #print("Train Scaled;", train_scaled)
            
            
            # Applying PCA 
            pca = PCA(n_components=9)
            train_pca = pca.fit_transform(train_scaled)
            #Concate RUL to new PCA Components
            df_train_pca = pd.DataFrame(train_pca, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9'])
            df_train = pd.concat([df_train_pca, Train_RUL,Train_Unit],axis = 1)

            Vald_Unit = df_vlad['unit'].reset_index(drop = True)
            Vald_Featuers = df_vlad.iloc[:,0:36].reset_index(drop = True)
            Vald_RUL = df_vlad['RUL'].reset_index(drop = True)

            
            # Applying Transform on Validation Dataset
            vlad_scaled = scaler.transform(Vald_Featuers)
            vald_pca = pca.transform(vlad_scaled)
            
            #Concate RUL to Validation_PCA
            df_vald_pca = pd.DataFrame(vald_pca, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9'])
            df_vlad = pd.concat([df_vald_pca, Vald_RUL,Vald_Unit], axis = 1)
            #print("Df_Vlad;", df_vlad)
         

    return df_train, df_vlad 



def df_test_creator(df_all, units_index_test):
    test_strata= []

    for idx in units_index_test:
        # selecting the training units
        df_test_temp = df_all[df_all['unit'] == np.float64(idx)]

        #Getting the number of datapoints in each RUL for unit 2
        count_values =  df_test_temp.value_counts('RUL', sort = False)

        # calculation for trainable samples for each RUL and unit

        count_values= list(count_values)
        #Sum of all samples in unit 2
        total_data_point_per_unit=sum(count_values)
       #Taking stratified random samples for each unit
        B=0
        sample_per_RUL=0
        for rul in count_values:
            #Getting required samples for each RUL
            sample_per_RUL_test=int(round(rul/total_data_point_per_unit*3550, 0 )) 
            df_test_temp_1 = df_test_temp[df_test_temp['RUL']==np.float64(B)]
            #print(df_train_temp_1)
            #Sampling randomly for each required rows per RUL for training
            df_test_temp_2 = df_test_temp_1.sample(n=sample_per_RUL_test,axis =0)
            #Getting the resultant of the dataframe for testing
            #df_vlad_test_temp_1 = df_train_temp_1.drop(df_train_temp_2.index)
            B=B+1
            
            test_strata.append(df_test_temp_2)
            df_test_unit_strata = pd.concat(test_strata)
            #Sorting the training dataframe in ascending order with respect to unit number, RUL number and alt number
            df_test = df_test_unit_strata.sort_values(by = ["unit","RUL","alt"], ascending=[True, False, True])
      
            Test_Unit = df_test['unit'].reset_index(drop = True)
            Test_Featuers = df_test.iloc[:,0:36].reset_index(drop = True)

            Test_RUL = df_test['RUL'].reset_index(drop = True)
            # Applying Transform on Test Dataset
            test_scaled = scaler.transform(Test_Featuers)
            
            # Applying PCA 
            test_pca = pca.transform(test_scaled)
            #Concate RUL to Test PCA
            df_test_pca = pd.DataFrame(test_pca, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9'])
            df_test = pd.concat([df_test_pca, Test_RUL,Test_Unit], axis = 1)

    return df_test


def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

def time_window_slicing (input_array, sequence_length, sequence_cols):
    # generate labels
    label_gen = [gen_labels(input_array[input_array['unit'] == id], sequence_length, ['RUL'])
                 for id in input_array['unit'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float64)
    # label_array = np.concatenate(label_gen)

    # transform each id of the train dataset in a sequence
    seq_gen = (list(gen_sequence(input_array[input_array['unit'] == id], sequence_length, sequence_cols))
               for id in input_array['unit'].unique())
    sample_array = np.concatenate(list(seq_gen)).astype(np.float64)
    # sample_array = np.concatenate(list(seq_gen))
    
    print("sample_array")

    
    return sample_array, label_array


def time_window_slicing_label_save (input_array, sequence_length, stride, index, sample_dir_path, sequence_cols = 'RUL'):
    '''
    ref
        for i in range(0, input_temp.shape[0] - sequence_length):
        window = input_temp[i*stride:i*stride + sequence_length, :]  # each individual window
        window_lst.append(window)
        # print (window.shape)


    '''
    # generate labels
    window_lst = []  
    input_temp = input_array[input_array['unit'] == index][sequence_cols].values
    num_samples = int((input_temp.shape[0] - sequence_length)/stride) + 1
    for i in range(num_samples):
        window = input_temp[i*stride:i*stride + sequence_length]  # each individual window
        window_lst.append(window)

    label_array = np.asarray(window_lst).astype(np.float64)



    return label_array[:,-1]

def time_window_slicing_sample_save (input_array, sequence_length, stride, index, sample_dir_path, sequence_cols):
    '''


    '''
    # generate labels
    window_lst = []  # a python list to hold the windows

    input_temp = input_array[input_array['unit'] == index][sequence_cols].values
    print ("Unit%s input array shape: " %index, input_temp.shape)
    num_samples = int((input_temp.shape[0] - sequence_length)/stride) + 1
    for i in range(num_samples):
        window = input_temp[i*stride:i*stride + sequence_length,:]  # each individual window
        window_lst.append(window)

    sample_array = np.dstack(window_lst).astype(np.float64)
    # sample_array = np.dstack(window_lst)
    print ("sample_array.shape", sample_array.shape)
    

    return sample_array



class Input_Gen(object):
    '''
    class for data preparation (sequence generator)
    '''

    def __init__(self, df_train,df_vlad, df_test, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                 unit_index, sampling, stride):
        '''

        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        print("the number of input signals: ", len(cols_normalize))
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        norm_df = pd.DataFrame(min_max_scaler.fit_transform(df_train[cols_normalize]),
                               columns=cols_normalize,
                               index=df_train.index)
        join_df = df_train[df_train.columns.difference(cols_normalize)].join(norm_df)
        df_train = join_df.reindex(columns=df_train.columns)


        norm_vlad_df = pd.DataFrame(min_max_scaler.fit_transform(df_vlad[cols_normalize]),
                               columns=cols_normalize,
                               index=df_vlad.index)
        vlad_join_df = df_vlad[df_vlad.columns.difference(cols_normalize)].join(norm_vlad_df)
        df_vlad = vlad_join_df.reindex(columns=df_vlad.columns)

        norm_test_df = pd.DataFrame(min_max_scaler.transform(df_test[cols_normalize]), columns=cols_normalize,
                                    index=df_test.index)
        test_join_df = df_test[df_test.columns.difference(cols_normalize)].join(norm_test_df)
        df_test = test_join_df.reindex(columns=df_test.columns)
        df_test = df_test.reset_index(drop=True)

        self.df_train = df_train
        self.df_vlad = df_vlad
        self.df_test = df_test

        print ("Self of train:", self.df_train)
        print ("Self of Vlad:", self.df_vlad)
        print ("Self of test:", self.df_test)

        self.cols_normalize = cols_normalize
        self.sequence_length = sequence_length
        self.sequence_cols = sequence_cols
        self.sample_dir_path = sample_dir_path
        self.unit_index = np.float64(unit_index)
        self.sampling = sampling
        self.stride = stride


    def seq_gen(self):
        '''
        concatenate vectors for NNs
        :param :
        :param :
        :return:
        '''

        if any(index == self.unit_index for index in self.df_train['unit'].unique()):
            print ("Unit for Train")
            label_array = time_window_slicing_label_save(self.df_train, self.sequence_length,
                                           self.stride, self.unit_index, self.sample_dir_path, sequence_cols='RUL')
            sample_array = time_window_slicing_sample_save(self.df_train, self.sequence_length,
                                           self.stride, self.unit_index, self.sample_dir_path, sequence_cols=self.cols_normalize)

        else:
         print("ERROR")

        print("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)



        np.savez_compressed(os.path.join(self.sample_dir_path, 'Train_Unit%s_win%s_str%s_smp%s' %(str(int(self.unit_index)), self.sequence_length, self.stride, self.sampling)),
                                         sample=sample_array, label=label_array)
        print ("Train unit saved")

        return

    def seq_gen_vlad(self):
        '''
        concatenate vectors for NNs
        :param :
        :param :
        :return:
        '''

        if any(index == self.unit_index for index in self.df_vlad['unit'].unique()):
            print ("Unit for vlad")
            label_array = time_window_slicing_label_save(self.df_vlad, self.sequence_length,
                                           self.stride, self.unit_index, self.sample_dir_path, sequence_cols='RUL')
            sample_array = time_window_slicing_sample_save(self.df_vlad, self.sequence_length,
                                           self.stride, self.unit_index, self.sample_dir_path, sequence_cols=self.cols_normalize)

        else:
         print("ERROR")

        print("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)



        np.savez_compressed(os.path.join(self.sample_dir_path, 'Vlad_Unit%s_win%s_str%s_smp%s' %(str(int(self.unit_index)), self.sequence_length, self.stride, self.sampling)),
                                         sample=sample_array, label=label_array)
        print ("Vlad unit saved")

        return

    def seq_gen_test(self):
        '''
        concatenate vectors for NNs
        :param :
        :param :
        :return:
        '''

        if any(index == self.unit_index for index in self.df_test['unit'].unique()):
            print("Unit for Test")
            label_array = time_window_slicing_label_save(self.df_test, self.sequence_length,
                                           self.stride, self.unit_index, self.sample_dir_path, sequence_cols='RUL')
            sample_array = time_window_slicing_sample_save(self.df_test, self.sequence_length,
                                           self.stride, self.unit_index, self.sample_dir_path, sequence_cols=self.cols_normalize)
        else:
            print("ERROR: ")
        print("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)

        np.savez_compressed(os.path.join(self.sample_dir_path, 'Test_Unit%s_win%s_str%s_smp%s' %(str(int(self.unit_index)), self.sequence_length, self.stride, self.sampling)),
                                         sample=sample_array, label=label_array)
        print ("test unit saved")

        return



