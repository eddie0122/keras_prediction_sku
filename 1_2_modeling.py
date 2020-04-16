### Libraries ##################################################################
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from datetime import datetime
import calendar

import matplotlib.pyplot as plt
import seaborn as sns

import math
from scipy import stats

import pickle as pkl

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras

### Dataset Load ###############################################################
### Import Dataset Dictionary
file_path = './data/dataset.pkl'
with open(file_path, 'rb') as output:
    dataset = pkl.load(output)

X_main = dataset.get('X1')
X_aux = dataset.get('X2')

y_main = dataset.get('y1')
y_aux = dataset.get('y2')

label = dataset.get('label')

### Split Input Dataset into 4 Pieces
col = ['store', 'weekday', 'month', 'day', 'day_rel', 'weekend', 'dayoff', 'holiday', 'customer', 'visit', 'transaction', 'sales', 'unit', 'col_avg_1', 'col_std_1', 'col_avg_2', 'col_std_2', 'col_avg_3', 'col_std_3', 'col_avg_4', 'col_std_4', 'col_avg_5', 'col_std_5', 'col_avg_6', 'col_std_6', 'col_avg_7', 'col_std_7', 'col_avg_8', 'col_std_8', 'col_avg_9', 'col_std_9', 'col_avg_10', 'col_std_10', 'col_avg_11', 'col_std_11', 'col_avg_12', 'col_std_12', 'col_avg_13', 'col_std_13', 'col_avg_14', 'col_std_14', 'col_avg_15', 'col_std_15', 'col_avg_16', 'col_std_16', 'col_avg_17', 'col_std_17', 'col_avg_18', 'col_std_18', 'col_avg_19', 'col_std_19', 'col_avg_20', 'col_std_20', 'col_avg_21', 'col_std_21', 'col_avg_22', 'col_std_22', 'col_avg_23', 'col_std_23', 'col_avg_24', 'col_std_24', 'col_avg_25', 'col_std_25', 'col_avg_26', 'col_std_26', 'col_avg_27', 'col_std_27', 'col_avg_28', 'col_std_28', 'col_avg_29', 'col_std_29', 'col_avg_30', 'col_std_30', 'col_avg_31', 'col_std_31', 'col_avg_32', 'col_std_32', 'col_avg_33', 'col_std_33', 'col_avg_34', 'col_std_34', 'col_avg_35', 'col_std_35', 'col_avg_36', 'col_std_36', 'col_avg_37', 'col_std_37', 'col_avg_38', 'col_std_38', 'col_avg_39', 'col_std_39', 'col_avg_40', 'col_std_40']

## Input 1 (customer, visit, transaction, sales, unit)
X_1 = X_main[:][:, :, 8:13]
X_1 = X_1.reshape(X_1.shape[0], X_1.shape[1], X_1.shape[2], 1)

## Input 2 (Engineered Feature : From col_avg_1 to col_std_40)
X_2 = X_main[:][:, :, 13:]

## Input 3 (store, weekday, month, day, day_rel, weekend, dayoff, holiday)
X_3 = X_main[:][:, :, :8]

## Input 4 (t+9's store, weekday, month, day, day_rel, weekend, dayoff, holiday)
X_4 = X_aux

### K-Fold (5) : Stratified by Stores
ix_start = label.groupby(['store_code']).head(1).index.tolist()
ix_end = label.groupby(['store_code']).tail(1).index.tolist()

idx_train = []
idx_test = []

k = 5
k_base = 75
k = int((100 - k_base) / 5)
for iters in range(k_base, 100, k):
    idx_train_tmp = []
    idx_test_tmp = []
    idx_val_tmp = []
    for ix1, ix2 in zip(ix_start, ix_end):
        ix_len1 = ix1 + math.ceil((ix2 - ix1) * iters / 100)
        ix_len2 = ix1 + math.ceil((ix2 - ix1) * (iters + k) / 100)

        v1 = list(range(ix1, ix_len1+1))
        v2 = list(range(ix_len1+1, ix_len2+1))
        idx_train_tmp += v1
        idx_test_tmp += v2

    idx_train.append(idx_train_tmp)
    idx_test.append(idx_test_tmp)

### Model ######################################################################
models = keras.models
layers = keras.layers
optimizers = keras.optimizers
callbacks = keras.callbacks
utils = keras.utils
K = keras.backend

### Model Create Function
def modelBuild(input_1, input_2, input_3, input_4, optm='adam', loss_wgt=[1, .2]):
    ### Model Block1
    input1_shape = input_1
    input1 = layers.Input(shape=input1_shape, name='B1_Input')
    h1 = layers.Conv2D(16, kernel_size=(2,2), activation='relu', name='B1_CNN1-1')(input1)
    h1 = layers.BatchNormalization(name='B1_CNN1-2')(h1)
    h1 = layers.Conv2D(1, kernel_size=(2,2), activation='relu', padding='same', name='B1_CNN1-3')(h1)
    h1 = layers.MaxPooling2D(pool_size=(2,1), name='B1_CNN1-4')(h1)
    h1 = layers.Reshape((h1.shape[1], h1.shape[2]))(h1)
    h1 = layers.LSTM(16, activation='tanh', name='B1_RNN1-1')(h1)

    ### Model Block2
    input2_shape = input_2
    input2 = layers.Input(shape=input2_shape, name='B2_Input')
    h2 = layers.LSTM(32, activation='tanh', name='B2_RNN1-2')(input2)

    ### Model Block3
    input3_shape = input_3
    input3 = layers.Input(shape=input3_shape, name='B3_Input')
    h3 = layers.LSTM(32, activation='tanh', name='B3_RNN1-1')(input3)

    ### Model Block4
    input4_shape = input_4
    input4 = layers.Input(shape=input4_shape, name='B4_Input')
    h4 = layers.Dense(4, activation='relu', name='B4_Dense1_1')(input4)
    h4 = layers.BatchNormalization(name='B4_Dense1_2')(h4)
    h4 = layers.Dense(16, activation='relu', name='B4_Dense1_3')(h4)
    h4 = layers.BatchNormalization(name='B4_Dense1_4')(h4)
    h4 = layers.Dense(4, activation='relu', name='B4_Dense1_5')(h4)

    ### Merge Block1 + Block3
    merge_1st = layers.concatenate([h1, h3], axis=1, name='Merge_1st')
    output_aux = layers.Dense(1, name='Output_AUX')(merge_1st)

    ### Merge Blocks
    merge_2nd = layers.concatenate([merge_1st, h2, h4], axis=1, name='Merge_2nd_Dense1-1')
    merge_2nd = layers.Dense(16, activation='relu', name='Merge_2nd_Dense1-2')(merge_2nd)
    merge_2nd = layers.BatchNormalization(name='Merge_2nd_Dense1-3')(merge_2nd)
    merge_2nd = layers.Dense(32, activation='relu', name='Merge_2nd_Dense1-4')(merge_2nd)
    merge_2nd = layers.BatchNormalization(name='Merge_2nd_Dense1-5')(merge_2nd)
    merge_2nd = layers.Dense(16, activation='relu', name='Merge_2nd_Dense1-6')(merge_2nd)
    output_main = layers.Dense(1, name='Output_Main')(merge_2nd)

    ### Model
    model_comb = models.Model([input1, input2, input3, input4], [output_main, output_aux])
    model_comb.compile(optimizer=optm, loss=['mse', 'mse'], loss_weights=loss_wgt)
    return model_comb


### Parameters
optimizer = optimizers.Adam() # Optimizer 
estop = callbacks.EarlyStopping(monitor='val_Output_Main_loss', patience=100) # Early Stop
n_batch = 512 * 8
n_epoch = 2000
val_ratio = 0.2

### 5-Fold Cross Validation (by Store Code)
model_hist = []
model_mse = []
model_mae = []
model_df = DataFrame()
iters = 1
for train, test in zip(idx_train, idx_test):
    model = modelBuild(X_1.shape[1:], X_2.shape[1:], X_3.shape[1:], 
                       X_4.shape[1:], optm=optimizer)
    
    hist = model.fit([X_1[train], X_2[train], X_3[train], X_4[train]], 
                     [y_main[train], y_aux[train]],
                     batch_size=n_batch, epochs=n_epoch, verbose=2,
                     callbacks=[estop], validation_split=val_ratio)
    model_hist.append(hist.history)

    y_true = y_main[test]
    y_pred = model.predict([X_1[test], X_2[test], X_3[test], X_4[test]])[0]
    
    mse = mean_squared_error(y_true, y_pred)
    model_mse.append(mse)
    
    mae = mean_absolute_error(y_true, y_pred)
    model_mae.append(mae)
    
    model_df_tmp = label.loc[test].copy()
    model_df_tmp['iter'] = iters
    model_df_tmp['y_true'] = y_true
    model_df_tmp['y_pred'] = y_pred
    model_df_tmp['mse'] = mse
    model_df_tmp['mae'] = mae
    model_df = model_df.append(model_df_tmp)

    iters += 1
    del model

### Export Result
data_export = {'history':model_hist, 'mse':model_mse, 'mae':model_mae, 'df':model_df}
file_path = './data/model_result.pkl'
with open(file_path, 'wb') as output:
    pkl.dump(data_export, output)


################################################################################
### Box-Cox
def boxCox(dataset, lam=0, inverse=False):
    if inverse:
        if lam == 0:
            v_return = math.exp(dataset)
        else:
            v_return = (dataset * lam + 1) ** (1 / lam)

    else:
        if lam == 0:
            v_return = np.log(dataset)
        else:
            v_return = (pow(dataset, lam) - 1) / lam
    
    return v_return

### Min-Max Scaler
file_path = './scaler/scaler_unit.pkl'
with open(file_path, 'rb') as output:
    scaler = pkl.load(output)

def inverseUnit(data, lam=.17):
    data = scaler.inverse_transform(data)
    data = boxCox(data, lam=lam, inverse=True)
    return data

model_df['y_true_actual'] = inverseUnit(model_df[['y_true']])
model_df['y_pred_actual'] = inverseUnit(model_df[['y_pred']])
model_df['y_pred_actual'].fillna(0, inplace=True)

### Lineplot by Stores
stores = model_df['store_code'].unique().tolist()
n_cols = 6
n_rows = math.ceil(len(stores) / n_cols)
ix = 0

fig, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
for n_row in range(n_rows):
    for n_col in range(n_cols):
        try:
            df_plt = model_df[model_df['store_code'] == stores[ix]].copy()
            df_plt = df_plt.reset_index(drop=True)
            ax[n_row, n_col].set_title("Store Code : {}".format(stores[ix]))
            ax[n_row, n_col].plot(df_plt['y_true'], color='blue')
            ax[n_row, n_col].plot(df_plt['y_pred'], color='red', alpha=.5)
        except:
            pass

        ix += 1
plt.show()

### Lineplot by Stores : Actual Value
stores = model_df['store_code'].unique().tolist()
n_cols = 6
n_rows = math.ceil(len(stores) / n_cols)
ix = 0

fig, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
for n_row in range(n_rows):
    for n_col in range(n_cols):
        try:
            df_plt = model_df[model_df['store_code'] == stores[ix]].copy()
            df_plt = df_plt.reset_index(drop=True)
            ax[n_row, n_col].set_title("Store Code : {}".format(stores[ix]))
            ax[n_row, n_col].plot(df_plt['y_true_actual'], color='blue')
            ax[n_row, n_col].plot(df_plt['y_pred_actual'], color='red', alpha=.5)
        except:
            pass

        ix += 1
plt.show()


###
model_df['mse'].unique()
model_df['mae'].unique()

mse_actual = []
mae_actual = []
for ix in range(1,6):
    df_tmp = model_df[model_df['iter'] == ix].copy()
    val1 = mean_squared_error(df_tmp['y_true_actual'], df_tmp['y_pred_actual'])
    val2 = mean_absolute_error(df_tmp['y_true_actual'], df_tmp['y_pred_actual'])

    mse_actual.append(val1)
    mae_actual.append(val2)





