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
import tensorflow as tf
seed = 0
random.seed(0)
np.random.seed(seed)

import keras.backend as K
import keras.backend as keras
from keras import backend
from keras import optimizers
from keras.models import Sequential, load_model, Model
from keras.layers import InputLayer, Dense, Flatten, Dropout, Embedding, InputLayer
from keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed, Bidirectional
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import GlorotNormal, GlorotUniform
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
initializer = tf.keras.initializers.GlorotNormal(seed=0)






def one_dcnn(n_filters, kernel_size, input_array, initializer):

    cnn = Sequential(name='one_d_cnn_2_Simple_F_10_10_K_5_5')
    cnn.add(InputLayer(input_shape=(input_array.shape[1],input_array.shape[2]), name = "Input"))
    cnn.add(Conv1D(filters=10, kernel_size=5,kernel_initializer=initializer,padding = 'same' , name = "Conv1D_1"))
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=10, kernel_size=5,kernel_initializer=initializer,padding = 'same',  name="Conv1D_2"))
    cnn.add(Activation('relu'))         
    cnn.add(Flatten())
    cnn.add(Dense(50, kernel_initializer=initializer, name="50_Connected_layer"))
    cnn.add(Activation('linear'))
    cnn.add(Dense(1, kernel_initializer=initializer))
    cnn.summary()
    return cnn

def mlps(vec_len, h1, h2, h3, h4, h5):
    """
    > This function creates a neural network with 4 hidden layers, each with a specified number of nodes
    
    :param vec_len: the length of the vector that will be fed into the model
    :param h1: number of neurons in the first hidden layer
    :param h2: number of neurons in the second hidden layer
    :param h3: number of hidden units in the third hidden layer
    :param h4: the number of neurons in the last hidden layer
    :param h5: the number of neurons in the last hidden layer
    :return: A model with the specified number of hidden layers and nodes.
    """
    '''
    '''

    model = Sequential()
    model.add(Dense(h1, activation='relu', input_shape=(vec_len,), name = "Input" ))
    model.add(Dense(h2, activation='relu'))
    model.add(Dense(h3, activation='relu'))
    model.add(Dense(h4, activation='relu'))
    model.add(Dense(h5, activation='relu'))
    model.add(Dense(1))
    return model
