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

from sklearn.preprocessing import MinMaxScaler
import pickle as pkl

### Load Data ##################################################################
file_path = folder + 'data_pandas.pkl'
df = pd.read_pickle(file_path)

### Transform Data #############################################################
### Box-Cox Transform Functions
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

def boxCoxPlot(dataset, col_name='sales', lam1=0, lam2=0.7):
    lambda_list = list(np.arange(lam1, lam2, 0.01))
    n_cols = 4
    n_rows = math.ceil(len(lambda_list) / n_cols)
    i = 0
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for n_row in range(n_rows):
        for n_col in range(n_cols):
            try:
                data_array = dataset.loc[dataset[col_name] > 0, col_name].copy()
                tmp = boxCox(data_array, lam=lambda_list[i]).ravel()
                tt = '{:.2f}, Skew {:.3f}, Kurt {:.3f}'.format(lambda_list[i], 
                                                            stats.skew(tmp),
                                                            stats.kurtosis(tmp))
                ax[n_row, n_col].set_title(tt)
                sns.distplot(tmp, kde=True, ax=ax[n_row, n_col])
            except:
                pass
            i += 1
    plt.show()

### Sales Distribution by Box-Cox Transform
boxCoxPlot(df, col_name='sales', lam1=0, lam2=.3)

### Customer Distribution by Box-Cox Transform
boxCoxPlot(df, col_name='customer', lam1=0, lam2=.3)

### Visit Distribution by Box-Cox Transform
boxCoxPlot(df, col_name='visit', lam1=0, lam2=.3)

### Transaction Distribution by Box-Cox Transform
boxCoxPlot(df, col_name='transaction', lam1=0, lam2=.3)

### Unit Distribution by Box-Cox Transform
boxCoxPlot(df, col_name='unit', lam1=0, lam2=.3)


### Transform & Scaling ########################################################
df_scaled = df[['store_code', 'date_id', 'week_group']].copy()

col_names = ['sales', 'customer', 'visit', 'transaction', 'unit']
col_lambda = [.27, .23, .25, .24, .24]
for ix1, ix2 in zip(col_names, col_lambda):
    df_scaled[ix1] = boxCox(df[ix1], ix2)


### Feature Engineering : Moving Variables (Average, STD) ######################
def movingFeature(dataset, col_name, wsize, groups=['store_code']):
    moving_mean = dataset.groupby(groups).rolling(wsize)[col_name].mean().values
    moving_std = dataset.groupby(groups).rolling(wsize)[col_name].std().values
    return moving_mean, moving_std

cols = ['store_code', 'week_group']
col_names = ['sales', 'customer', 'visit', 'transaction', 'unit']
winsize = range(3, 11); ix = 0
for ix1 in col_names:
    for ix2 in winsize:
        ix_col1 = 'col{}_avg'.format(ix)
        ix_col2= 'col{}_std'.format(ix)
        df_scaled[ix_col1], df_scaled[ix_col2] = \
            movingFeature(df, col_name=ix1, wsize=ix2, groups=cols)
        ix += 1
df_scaled = df_scaled.dropna().reset_index(drop=True)


### Non-Metric Feature Scaling #################################################
## WeekDays
df_scaled['weekday'] = df_scaled['date_id'].apply(lambda x: 
                                                  datetime.isoweekday(x))

## Month
df_scaled['month'] = df_scaled['date_id'].apply(lambda x: x.month) / 12

## Day
df_scaled['day'] = df_scaled['date_id'].apply(lambda x: x.day) / 31

## Day (Relative : Divided by days of month)
df_scaled['day_rel'] = df_scaled['date_id']\
    .apply(lambda x: x.day / calendar.monthrange(x.year, x.month)[-1])

## Weekend : Weekday(0) & Saturday(1) & Sunday(2)
def weekEnd(dataset):
    if dataset == 6:
        v_return = 1
    elif dataset == 7:
        v_return = 2
    else:
        v_return = 0
    return v_return

df_scaled['weekend'] = df_scaled['weekday'].apply(lambda x: weekEnd(x))
df_scaled['weekend'] = df_scaled['weekend'] / 2

## WeekDays
df_scaled['weekday'] = df_scaled['date_id']\
    .apply(lambda x: datetime.isoweekday(x)) / 7

## Store Code
df_scaled['store_code'] = df_scaled['store_code'].astype(int)
df_scaled['store_code'] = df_scaled['store_code'] / 5000


### Scaling for Neural Network Model ###########################################
scaler_unit = MinMaxScaler() # Scaler for Unit
scaler_totl = MinMaxScaler() # scaler for all numeric features

## Scaler for Unit
scaler_unit.fit(df_scaled[['unit']])

## Scaler for All Numeric features (with Transform)
col = df_scaled.columns.tolist()[3:-5]
df_scaled[col] = scaler_totl.fit_transform(df_scaled[col])


### Create Dataset #############################################################
col1 = df_scaled.columns.tolist()[3:8]
col2 = df_scaled.columns.tolist()[8:-5]
col3 = ['store_code', 'weekday', 'month', 'day', 'day_rel', 'weekend']
cols = col1 + col2 + col3

winsize = 21 # Windows size of past period
step = 9 # 9 days in advance
step_aux = 1 # 1 day in advance
X, y, y_aux = [], [], []

store_list = df_scaled['store_code'].unique().tolist()
for store in store_list:
    df_slice1 = df_scaled[df_scaled['store_code'] == store].copy()
    week_list = df_slice1['week_group'].unique().tolist()
    for week in week_list:
        df_slice2 = df_slice1[df_slice1['week_group'] == week].copy()
        df_slice2 = df_slice2.reset_index(drop=True)
        for idx in range(len(df_slice2) - winsize - step + 1):
            array_data = df_slice2.loc[idx:idx + winsize - 1, cols].values
            array_label = df_slice2.loc[idx + winsize + step - 1, 'unit']
            array_label_aux = df_slice2.loc[idx + winsize + step_aux - 1, 'unit']
            X.append(array_data)
            y.append(array_label)
            y_aux.append(array_label_aux)
X = np.array(X) # Dataset
y = np.array(y) # Main Label Set
y_aux = np.array(y_aux) # Auxiliary Label Set


### Split Dataset into 3 Pieces ################################################
## Input 1
X_input1 = X[:][:, :, :5]
X_input1 = X_input1.reshape(X_input1.shape[0], X_input1.shape[1], 
                            X_input1.shape[2], 1)
## Input2
X_input2 = X[:][:, :, 5:-6]

## input3
X_input3 = X[:][:, :, -6:]


### Export Scalers & Dataset ###################################################
## Make Dictionary Dataset
dataset_export = {'X1':X_input1, 'X2':X_input2, 'X3':X_input3, 
                  'y_main': y, 'y_aux': y_aux}

## Export Dataset
with open('{}dataset.pkl'.format(folder), 'wb') as output:
    pkl.dump(dataset_export, output)

## Export Box-Cox Transform & Scalers
for ix1, ix2 in zip([scaler_unit, scaler_totl, boxCox], 
                    ['scaler_unit', 'scaler_totl', 'scaler_boxcox']):
    with open('{}{}.pkl'.format(folder, ix2), 'wb') as output:
        pkl.dump(ix1, output)
