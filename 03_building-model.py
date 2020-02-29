### Libraries ##################################################################
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from datetime import datetime, date, time, timedelta
import calendar

import matplotlib.pyplot as plt
import seaborn as sns

import os
folder = './ml_predict/'
try:
    os.mkdir(folder)
except:
    pass

import math
from scipy import stats

import pickle as pkl

from sklearn.model_selection import KFold, cross_validate

### Load Dataset ###############################################################
with open('{}{}'.format(folder, 'dataset.pkl'), 'rb') as output:
    dataset = pkl.load(output)

X_input1 = dataset.get('X1')
X_input2 = dataset.get('X2')
X_input3 = dataset.get('X3')

y = dataset.get('y_main')
y_aux = dataset.get('y_aux')


### Neural Network Model #######################################################
import tensorflow as tf
from tensorflow import keras

models = keras.models
layers = keras.layers
optimizers = keras.optimizers
callbacks = keras.callbacks
utils = keras.utils
K = keras.backend

### R-Sqaured Metric for Model Evaluation
def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

### Model Create Function
def modelCreate(input_one, input_two, input_three, optm='adam', loss_wgt=[1, .2]):
    ### Model Block1
    input1_shape = input_one
    input1 = layers.Input(shape=input1_shape, name='B1_Input')
    h1 = layers.Conv2D(16, kernel_size=(2,2), activation='relu', name='B1_CNN1-1')(input1)
    h1 = layers.BatchNormalization(name='B1_CNN1-2')(h1)
    h1 = layers.Conv2D(1, kernel_size=(2,2), activation='relu', padding='same', name='B1_CNN1-3')(h1)
    h1 = layers.MaxPooling2D(pool_size=(2,1), name='B1_CNN1-4')(h1)
    h1 = layers.Reshape((h1.shape[1], h1.shape[2]))(h1)
    h1 = layers.LSTM(16, activation='tanh', name='B1_RNN1-1')(h1)


    ### Model Block2
    input2_shape = input_two
    input2 = layers.Input(shape=input2_shape, name='B2_Input')
    h2 = layers.LSTM(32, activation='tanh', name='B2_RNN1-2')(input2)


    ### Model Block3
    input3_shape = input_three
    input3 = layers.Input(shape=input3_shape, name='B3_Input')
    h3 = layers.LSTM(32, activation='tanh', name='B3_RNN1-1')(input3)

    ### Merge Block1 + Block3
    merge_1st = layers.concatenate([h1, h3], axis=1, name='Merge_1st')
    output_aux = layers.Dense(1, name='Output_AUX')(merge_1st)

    ### Merge Blocks
    merge_2nd = layers.concatenate([merge_1st, h2], axis=1, name='Merge_2nd_Dense1-1')
    merge_2nd = layers.Dense(16, activation='relu', name='Merge_2nd_Dense1-2')(merge_2nd)
    merge_2nd = layers.BatchNormalization(name='Merge_2nd_Dense1-3')(merge_2nd)
    merge_2nd = layers.Dense(32, activation='relu', name='Merge_2nd_Dense1-4')(merge_2nd)
    merge_2nd = layers.BatchNormalization(name='Merge_2nd_Dense1-5')(merge_2nd)
    merge_2nd = layers.Dense(16, activation='relu', name='Merge_2nd_Dense1-6')(merge_2nd)
    output_main = layers.Dense(1, name='Output_Main')(merge_2nd)

    ### Model
    model_comb = models.Model([input1, input2, input3], [output_main, output_aux])
    model_comb.compile(optimizer=optm, loss=['mse', 'mse'], 
                       metrics=[r2_keras, r2_keras], loss_weights=loss_wgt)
    return model_comb

### Parameters
optimizer = optimizers.Adam() # Optimizer
estop = callbacks.EarlyStopping(monitor='val_loss', patience=100) # Early Stop
n_batch = 512 * 4
n_epoch = 2000
val_ratio = 0.25

### Cross Validation
cv = KFold(n_splits=5, shuffle=True, random_state=0)
model_history, model_cv = [], []
for train, test in cv.split(X_input1):
    model_loop = modelCreate(X_input1.shape[1:], X_input2.shape[1:], 
                             X_input3.shape[1:], optm=optimizer, loss_wgt=[1, 0.2])
    
    hist = model_loop.fit([X_input1[train], X_input2[train], X_input3[train]], 
                          [y[train], y_aux[train]],
                          batch_size=n_batch, epochs=n_epoch, callbacks=[estop],
                          validation_split=val_ratio, verbose=2)

    eva = model_loop.evaluate([X_input1[test], X_input2[test], X_input3[test]], 
                              [y[test], y_aux[test]], batch_size=n_batch)

    model_history.append(hist.history)
    model_cv.append(eva)

    del model_loop
