################################################################################
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from datetime import datetime
import calendar

import matplotlib.pyplot as plt
import seaborn as sns

import os
import math
from scipy import stats
import pickle as pkl

from sklearn.preprocessing import MinMaxScaler
################################################################################
### Load Data ##################################################################
file_path = './data/data_product.pkl' # 2016-07-01 - 2020-02-29
df_raw = pd.read_pickle(file_path)
df_raw['date_id'] = pd.to_datetime(df_raw['date_id'])
df = df_raw.loc[(df_raw['prod_code'] == '2000100548807') & 
                (df_raw['date_id'] >= '2017-01-01')].copy()

### Week Number
df['week_num'] = df['date_id'].apply(lambda x: datetime.isocalendar(x)[0]*100 + 
                                     datetime.isocalendar(x)[1])

### Week Order Dictionary
week_numb = list(df['week_num'].sort_values().unique())
_df = DataFrame({'week_num':week_numb, 'week_order':range(len(week_numb))})
week_dict = _df.set_index('week_num').to_dict()['week_order']
del _df, week_numb

### Apply Week Order on Transaction Data
df['week_order'] = df['week_num'].apply(lambda x: week_dict.get(x))

### Find Streak Time-Series Group
cols = ['store_code', 'week_num']
df_week = df.groupby(cols).agg({'date_id':'count'}).reset_index()
df_week = df_week[df_week['date_id'] >= 4]
df_week['week_order'] = df_week['week_num'].apply(lambda x: week_dict.get(x))

_df = df_week['week_order'] - df_week\
    .groupby(['store_code'])['week_order'].shift(1)
def orderWeek(dataset):
    if (dataset == 1) | (dataset == 0):
        v_return = 0
    else:
        v_return = 1
    return v_return
_df = _df.fillna(0).apply(lambda x: orderWeek(x))

df_week['week_order1'] = _df
df_week['week_group'] = df_week\
    .groupby(['store_code'])['week_order1'].transform(lambda x: x.cumsum())

cols = ['store_code', 'week_group']
df_week_agg = df_week.groupby(cols).agg({'week_order':'count'}).reset_index()
df_week_agg = df_week_agg.loc[df_week_agg['week_order'] >= 5]
df_week_agg.drop(['week_order'], axis=1, inplace=True)

df_week = df_week.merge(df_week_agg, on=cols, how='inner')
df_week = df_week[['store_code', 'week_order','week_group']]

df = df.merge(df_week, on=['store_code', 'week_order'], how='inner')
del df_week, df_week_agg, _df

### Fill & Pad Full Date Records
def dateFill(start='2019-01-01', 
             end='2019-02-28', code1='3000', code2='4000', code3='5000'):
    df_inner_tmp = DataFrame({'store_code':code1, 'prod_code':code2, 
                              'week_group': code3, 
                              'date_id': pd.date_range(start, end, freq='D')})
    return df_inner_tmp

cols = ['store_code', 'prod_code', 'week_group']
calc = ['min', 'max']
df_week_list = df.groupby(cols).agg({'date_id':calc}).reset_index()

df_date = DataFrame()
for idx in df_week_list.index:
    v_start = df_week_list.loc[idx, 'date_id']['min']
    v_end = df_week_list.loc[idx, 'date_id']['max']
    
    v_store = df_week_list.loc[idx, 'store_code'][0]
    v_prod = df_week_list.loc[idx, 'prod_code'][0]
    v_group = df_week_list.loc[idx, 'week_group'][0]

    df_date_tmp = dateFill(v_start, v_end, v_store, v_prod, v_group)
    df_date = df_date.append(df_date_tmp)

cols = ['store_code', 'prod_code', 'week_group', 'date_id']
df_date_idx = df_date.set_index(cols)
df_idx = df.set_index(cols)

### Padding Full Date Data
df = df_idx.join(df_date_idx, how='outer').reset_index()
df['prod_desc'].fillna(method='ffill', inplace=True)

cols = ['customer', 'visit', 'transaction', 'sales', 'unit']
df[cols] = df[cols].fillna(0)

df.drop(['week_num', 'week_order'], axis=1, inplace=True)

### Dayoff Data (DIDA)
file_path = './data/data_dayoff.pkl'
df_dayoff = pd.read_pickle(file_path)
df_dayoff.drop(['format_code', 'fis_week_id'], axis=1, inplace=True)

df_dayoff.rename(columns={'DIDA_date':'date_id'}, inplace=True)
df_dayoff['date_id'] = pd.to_datetime(df_dayoff['date_id'])
df_dayoff['dayoff'] = 1

df = df.merge(df_dayoff, on=['store_code', 'date_id'], how='left')
df['dayoff'] = df['dayoff'].fillna(0).astype(int)

### Holiday
file_path = './data/data_holiday.csv'
df_holiday = pd.read_csv(file_path, dtype=str)
df_holiday['date_id'] = pd.to_datetime(df_holiday['date_id'])
df_holiday['holiday'] = 1
df_holiday.drop(['holiday_desc', 'holiday_type'], axis=1, inplace=True)
df_holiday.drop_duplicates(['date_id'], inplace=True)

df = df.merge(df_holiday, on=['date_id'], how='left')
df['holiday'] = df['holiday'].fillna(0).astype(int)


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

### Distribution Plot by Box-Cox with Various Lambda
# boxCoxPlot(df, col_name='sales', lam2=.4)
# boxCoxPlot(df, col_name='customer', lam2=.4)
# boxCoxPlot(df, col_name='visit', lam2=.4)
# boxCoxPlot(df, col_name='transaction', lam2=.4)
# boxCoxPlot(df, col_name='unit', lam2=.4)

### Transform
df_scale = df.copy()
col_lambda = [.18, .16, .16, .17, .17]
cols = ['sales', 'customer', 'visit', 'transaction', 'unit']
for ix1, ix2 in zip(cols, col_lambda):
    df_scale[ix1] = boxCox(df_scale[ix1], ix2)


### Feature Engineering : Moving Variables
def movingFeature(dataset, col_name, wsize, groups=['store_code']):
    moving_mean = dataset.groupby(groups).rolling(wsize)[col_name].mean().values
    moving_std = dataset.groupby(groups).rolling(wsize)[col_name].std().values
    return moving_mean, moving_std

cols = ['store_code', 'prod_code', 'week_group']
col_names = ['sales', 'customer', 'visit', 'transaction', 'unit']
winsize = range(3, 11)
ix = 1
for ix1 in col_names:
    for ix2 in winsize:
        col1 = 'col_avg_{}'.format(ix)
        col2 = 'col_std_{}'.format(ix)
        df_scale[col1], df_scale[col2] = movingFeature(df_scale, col_name=ix1, 
                                                       wsize=ix2, groups=cols)
        ix += 1

df_scale = df_scale.dropna().reset_index(drop=True)


### Non-Metric Feature Scaling
## WeekDays
def dateFeature(dataset):
    dataset['weekday'] = dataset['date_id']\
        .apply(lambda x: datetime.isoweekday(x))
    dataset['month'] = dataset['date_id']\
        .apply(lambda x: x.month) / 12 # Month
    dataset['day'] = dataset['date_id']\
        .apply(lambda x: x.day) / 31 # Day
    dataset['day_rel'] = dataset['date_id']\
        .apply(lambda x: x.day / calendar.monthrange(x.year, x.month)[-1]) # Day (Relative : Divided by days of month)

    ## Weekend : Weekday(0) & Saturday(1) & Sunday(2)
    def weekEnd(dataset):
        if dataset == 6:
            v_return = 1
        elif dataset == 7:
            v_return = 2
        else:
            v_return = 0
        return v_return
    dataset['weekend'] = dataset['weekday'].apply(lambda x: weekEnd(x))
    dataset['weekend'] = dataset['weekend'] / 2

    ## WeekDays
    dataset['weekday'] = dataset['date_id']\
        .apply(lambda x: datetime.isoweekday(x)) / 7

    ## Store Code
    dataset['store'] = dataset['store_code'].astype(int) / 5000

    return dataset

df_scale = dateFeature(df_scale)

### Scaling for Neural Network Model ###########################################
scaler_unit = MinMaxScaler() # Scaler for Unit
scaler_totl = MinMaxScaler() # scaler for all numeric features

## Export Scaler for Unit
scaler_unit.fit(df_scale[['unit']])
file_path = './scaler/scaler_unit.pkl'
with open(file_path, 'wb') as output:
    pkl.dump(scaler_unit, output)

## Export Scaler for All Numeric Features
col = list(df_scale.columns[5:10]) + list(df_scale.columns[12:-6])
df_scale[col] = scaler_totl.fit_transform(df_scale[col])

file_path = './scaler/scaler_totl.pkl'
with open(file_path, 'wb') as output:
    pkl.dump(scaler_totl, output)

################################################################################
### Slice Data
col_1 = ['store', 'weekday', 'month', 'day', 'day_rel', 'weekend', 'dayoff', 
         'holiday'] # Non-Numeric Features
col_2 = list(df_scale.columns[5:10]) + list(df_scale.columns[12:-6]) # Numeric Features
col = col_1 + col_2
col_label = ['store_code', 'date_id', 'dayoff', 'holiday']

window = 7 * 3
step_main = 9
step_aux = 1

X_main = [] # Dataset from t-21 to t-0 (All Features)
X_aux = [] # Dataset t+9 (Non-Numeric Features)
y_main = [] # Unit t+9
y_aux = [] # Unit t+1
y_label = pd.DataFrame() # Basic data t+9 (store_code, date_id, dayoff, holiday)

store_list = df_scale['store_code'].unique().tolist()
for order, store in enumerate(store_list):
    df_temp = df_scale[df_scale['store_code'] == store].copy()
    week_list = df_temp['week_group'].unique().tolist()

    for week in week_list:
        df_slice = df_temp[df_temp['week_group'] == week].copy()
        df_slice = df_slice.reset_index(drop=True)

        for ix in range(len(df_slice) - window - step_main + 1):
            _X_main = df_slice.loc[ix:ix + window - 1, col].values # X_main dataset
            _X_aux = df_slice.loc[ix + window + step_main - 1:ix + window + step_main, col_1].values[0] # X_main dataset
            
            _y_main = df_slice.loc[ix + window + step_main - 1, 'unit'] # y_main dataset
            _y_aux = df_slice.loc[ix + window + step_aux - 1, 'unit'] # y_aux dataset
            
            _y_label = df_slice.loc[ix + window + step_main - 1, col_label] # X_label
            
            X_main.append(_X_main); X_aux.append(_X_aux)
            y_main.append(_y_main); y_aux.append(_y_aux)

            y_label = y_label.append(_y_label)
    print('{} - {} / {}'.format(store, order+1, len(store_list)))

X_main = np.array(X_main); X_aux = np.array(X_aux)
y_main = np.array(y_main); y_aux = np.array(y_aux)
y_label = y_label.reset_index(drop=True)

### Dataset Dictionary
dataset = {'X1':X_main, 'X2':X_aux, 'y1':y_main, 'y2':y_aux, 'label':y_label}

### Export Dataset Dictionary
file_path = './data/dataset.pkl'
with open(file_path, 'wb') as output:
    pkl.dump(dataset, output)
