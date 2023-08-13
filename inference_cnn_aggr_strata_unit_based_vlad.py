## Import libraries in python
import gc
import argparse
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
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import random
from random import shuffle
from tqdm.keras import TqdmCallback

seed = 0
random.seed(0)
np.random.seed(seed)

import importlib
from scipy.stats import randint, expon, uniform
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
import scipy.stats as stats
import tensorflow as tf
print(tf.__version__)
import keras.backend as K
from keras import backend
from keras import optimizers
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Flatten, Dropout, Embedding
from keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed, Bidirectional
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

from keras.initializers import GlorotNormal, GlorotUniform

initializer = GlorotNormal(seed=0)

from utils.dnn_updated_arch import one_dcnn

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')

model_temp_path = os.path.join(current_dir, 'Models', 'oned_cnn_rep.h5')
tf_temp_path = os.path.join(current_dir, 'TF_Model_tf')

pic_dir = os.path.join(current_dir, 'Figures')

'''
load array from npz files that is created using the sample_creator_strata_unit
'''
def load_part_array (sample_dir_path, unit_num, win_len, stride, part_num):
    filename =  'Train_Unit%s_win%s_str%s_part%s.npz' %(str(int(unit_num)), win_len, stride, part_num)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)
    return loaded['sample'], loaded['label']

def load_part_array_vlad(sample_dir_path, unit_num, win_len, stride, part_num):
    filename =  'Vlad_Unit%s_win%s_str%s_part%s.npz' %(str(int(unit_num)), win_len, stride, part_num)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)
    return loaded['sample'], loaded['label']

def load_part_array_merge(sample_dir_path, unit_num, win_len, win_stride, partition):
    sample_array_lst = []
    label_array_lst = []
    print ("Unit: ", unit_num)
    for part in range(partition):
      print ("Part.", part+1)
      sample_array, label_array = load_part_array (sample_dir_path, unit_num, win_len, win_stride, part+1)
      sample_array_lst.append(sample_array)
      label_array_lst.append(label_array)
    sample_array = np.dstack(sample_array_lst)
    label_array = np.concatenate(label_array_lst)
    sample_array = sample_array.transpose(2, 0, 1)
    print ("sample_array.shape", sample_array.shape)
    print ("label_array.shape", label_array.shape)
    return sample_array, label_array

def load_part_array_merge_vlad(sample_dir_path, unit_num, win_len, win_stride, partition):
    sample_array_lst = []
    label_array_lst = []
    print ("Unit: ", unit_num)
    for part in range(partition):
      print ("Part.", part+1)
      sample_array, label_array = load_part_array_vlad(sample_dir_path, unit_num, win_len, win_stride, part+1)
      sample_array_lst.append(sample_array)
      label_array_lst.append(label_array)
    sample_array = np.dstack(sample_array_lst)
    label_array_vlad = np.concatenate(label_array_lst)
    sample_array_vlad = sample_array.transpose(2, 0, 1)
    print ("sample_array.shape", sample_array.shape)
    print ("label_array.shape", label_array.shape)
    return sample_array_vlad, label_array_vlad

def load_part_array_test(sample_dir_path, unit_num, win_len, stride, part_num):
    filename =  'Test_Unit%s_win%s_str%s_part%s.npz' %(str(int(unit_num)), win_len, stride, part_num)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)
    return loaded['sample'], loaded['label']

def load_part_array_merge_test(sample_dir_path, unit_num, win_len, win_stride, partition):
    sample_array_lst = []
    label_array_lst = []
    print ("Unit: ", unit_num)
    for part in range(partition):
      print ("Part.", part+1)
      sample_array, label_array = load_part_array_test(sample_dir_path, unit_num, win_len, win_stride, part+1)
      sample_array_lst.append(sample_array)
      label_array_lst.append(label_array)
    sample_array = np.dstack(sample_array_lst)
    label_array = np.concatenate(label_array_lst)
    sample_array = sample_array.transpose(2, 0, 1)
    print ("sample_array.shape", sample_array.shape)
    print ("label_array.shape", label_array.shape)
    return sample_array, label_array

def load_array(sample_dir_path, unit_num, win_len, stride, sampling):
    filename =  'Train_Unit%s_win%s_str%s_smp%s.npz' %(str(int(unit_num)), win_len, stride, sampling)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']

def load_array_vlad(sample_dir_path, unit_num, win_len, stride, sampling):
    filename =  'Vlad_Unit%s_win%s_str%s_smp%s.npz' %(str(int(unit_num)), win_len, stride, sampling)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']

def load_array_test(sample_dir_path, unit_num, win_len, stride, sampling):
    filename =  'Test_Unit%s_win%s_str%s_smp%s.npz' %(str(int(unit_num)), win_len, stride, sampling)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def figsave(history, win_len, win_stride, bs, lr, sub):
    fig_acc = plt.figure(figsize=(15, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training', fontsize=24)
    plt.ylabel('loss', fontdict={'fontsize': 18})
    plt.xlabel('epoch', fontdict={'fontsize': 18})
    plt.legend(['Training loss', 'Validation loss'], loc='upper left', fontsize=18)
    plt.show()
    print ("saving file:training loss figure")
    fig_acc.savefig(pic_dir + "/training_w%s_s%s_bs%s_sub%s_lr%s.png" %(int(win_len), int(win_stride), int(bs), int(sub), str(lr)))
    return



def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


def scheduler(epoch, lr):
    if epoch == 30:
        print("lr decay by 10")
        return lr * 0.1
    elif epoch == 70:
        print("lr decay by 10")
        return lr * 0.1
    else:
        return lr


def release_list(a):
   del a[:]
   del a
units_index_train = [2.0, 5.0, 10.0,11.0,14.0,15.0,16.0,18.0,20.0]
units_index_vlad = [2.0, 5.0, 10.0,11.0,14.0,15.0,16.0,18.0,20.0]
units_index_test = [2.0, 5.0, 10.0,11.0,14.0,15.0,16.0,18.0,20.0]

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=50, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-f', type=int, default=10, help='number of filter')
    parser.add_argument('-k', type=int, default=10, help='size of kernel')
    parser.add_argument('-bs', type=int, default=256, help='batch size')
    parser.add_argument('-ep', type=int, default=30, help='max epoch')
    parser.add_argument('-pt', type=int, default=20, help='patience')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-sub', type=int, default=10, help='subsampling stride')
    parser.add_argument('--sampling', type=int, default=1, help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')
    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s
    partition = 3
    n_filters = args.f
    kernel_size = args.k
    lr = args.lr
    bs = args.bs
    ep = args.ep
    pt = args.pt
    # vs = args.vs
    sub = args.sub
    sampling = args.sampling

    amsgrad = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True, name='Adam')
    rmsop = optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
                               name='RMSprop')

    train_units_samples_lst =[]
    train_units_labels_lst = []

    for index in units_index_train:
        print("Load data index: ", index)
        sample_array, label_array = load_array (sample_dir_path, index, win_len, win_stride, sampling)
        sample_array, label_array = shuffle_array(sample_array, label_array)
        print("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)
        sample_array = sample_array[::sub]
        label_array = label_array[::sub]
        print("sub sample_array.shape", sample_array.shape)
        print("sub label_array.shape", label_array.shape)
        train_units_samples_lst.append(sample_array)
        train_units_labels_lst.append(label_array)

    sample_array = np.concatenate(train_units_samples_lst)
    label_array = np.concatenate(train_units_labels_lst)
    print ("samples are aggregated")

    release_list(train_units_samples_lst)
    release_list(train_units_labels_lst)
    train_units_samples_lst =[]
    train_units_labels_lst = []
    print("Memory released")

    sample_array, label_array = shuffle_array(sample_array, label_array)
    print("samples are shuffled")
    print("sample_array.shape", sample_array.shape)
    print("label_array.shape", label_array.shape)

    print ("train sample dtype", sample_array.dtype)
    print("train label dtype", label_array.dtype)

  
    vlad_units_samples_lst =[]
    vlad_units_labels_lst =[]

    for index in units_index_vlad:
        print("Load data index: ", index)
        sample_array_vlad, label_array_vlad = load_array_vlad(sample_dir_path, index, win_len, win_stride, sampling)
     
        print("sample_array_vlad.shape", sample_array_vlad.shape)
        print("label_array_vlad.shape", label_array_vlad.shape)
        sample_array_vlad = sample_array_vlad[::sub]
        label_array_vlad = label_array_vlad[::sub]
        print("sub sample_array_vlad.shape", sample_array_vlad.shape)
        print("sub label_array_vlad.shape", label_array_vlad.shape)
        vlad_units_samples_lst.append(sample_array_vlad)
        vlad_units_labels_lst.append(label_array_vlad)

    sample_array_vlad = np.concatenate(vlad_units_samples_lst)
    label_array_vlad = np.concatenate(vlad_units_labels_lst)
    print ("samples_vlad are aggregated")

    release_list(vlad_units_samples_lst)
    release_list(vlad_units_labels_lst)
    vlad_units_samples_lst =[]
    vlad_units_labels_lst = []
    print("Memory released")


    print("sample_array_vlad.shape", sample_array_vlad.shape)
    print("label_array_vlad.shape", label_array_vlad.shape)

    print ("Vlad  sample dtype", sample_array_vlad.dtype)
    print("Vlad label dtype", label_array_vlad.dtype)

    one_d_cnn_model = one_dcnn(n_filters, kernel_size, sample_array, initializer)
    print(one_d_cnn_model.summary())
    
    start = time.time()
    lr_scheduler = LearningRateScheduler(scheduler)

    one_d_cnn_model.compile(loss='mean_squared_error', optimizer=amsgrad, metrics='mae')
    history = one_d_cnn_model.fit(sample_array, label_array, epochs=ep, batch_size=bs, validation_data=(sample_array_vlad, label_array_vlad), verbose=2,
                      callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=pt, verbose=1, mode='min'),
                                    ModelCheckpoint(model_temp_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)]
                      )
    figsave(history, win_len, win_stride, bs, lr, sub)

    print("The FLOPs is:{}".format(get_flops(one_d_cnn_model)), flush=True)
    num_train = sample_array.shape[0]
    num_vlad = sample_array_vlad.shape[0]
    end = time.time()
    training_time = end - start
    print("Training time: ", training_time)

    ### Test (inference after training)
    start = time.time()
    output_lst = []
    truth_lst = []

    for index in units_index_test:
        print ("test idx: ", index)
        sample_array, label_array = load_array_test(sample_dir_path, index, win_len, win_stride, sampling)
        # estimator = load_model(tf_temp_path, custom_objects={'rmse':rmse})
        print("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)
        sample_array = sample_array[::sub]
        label_array = label_array[::sub]
        print("sub sample_array.shape", sample_array.shape)
        print("sub label_array.shape", label_array.shape)

        estimator = load_model(model_temp_path)
        y_pred_test = estimator.predict(sample_array)
        output_lst.append(y_pred_test)
        truth_lst.append(label_array)

    print(output_lst[0].shape)
    print(truth_lst[0].shape)

    print(np.concatenate(output_lst).shape)
    print(np.concatenate(truth_lst).shape)

    output_array = np.concatenate(output_lst)[:, 0]
    trytg_array = np.concatenate(truth_lst)
    print(output_array.shape)
    print(trytg_array.shape)
    rms = sqrt(mean_squared_error(output_array, trytg_array))
    print(rms)
    rms = round(rms, 2)

    end = time.time()
    inference_time = end - start
    num_test = output_array.shape[0]

    for idx in range(len(units_index_test)):
        print(output_lst[idx])
        print(truth_lst[idx])
        fig_verify = plt.figure(figsize=(24, 10))
        plt.plot(truth_lst[idx], color="red", linewidth = 1.0)
        plt.plot(output_lst[idx], color="black", linewidth = 1.0)
        plt.title('Unit %s Testing' %str(int(units_index_test[idx])), fontsize=30)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('RUL', fontdict={'fontsize': 24})
        plt.xlabel('Timestamps', fontdict={'fontsize': 24})
        plt.legend(['Truth', 'Predicted'], loc='upper right', fontsize=28)
        plt.show()
        fig_verify.savefig(pic_dir + "/unit%s_test_w%s_s%s_bs%s_lr%s_sub%s_rmse-%s.png" %(str(int(units_index_test[idx])),
                                                                              int(win_len), int(win_stride), int(bs),
                                                                                    str(lr), int(sub), str(rms)))

    print("The FLOPs is:{}".format(get_flops(one_d_cnn_model)), flush=True)
    print("wind length_%s,  win stride_%s" %(str(win_len), str(win_stride)))
    print("# Training samples: ", num_train)
    print("# Validation samples: ", num_vlad)
    print("# Inference samples: ", num_test)
    print("Training time: ", training_time)
    print("Inference time: ", inference_time)
    print("Result in RMSE: ", rms)


if __name__ == '__main__':
    main()
