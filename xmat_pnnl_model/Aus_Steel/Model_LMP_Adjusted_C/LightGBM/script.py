#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: script.py
#AUTHOR: Osman Mamun
#DATE CREATED: 12-19-2019

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xmat_pnnl_code import ProcessData
from xmat_pnnl_code import GBM

#Model data
path = '/Users/mamu867/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data/LMP'
model = np.load(path + '/model_params.npy', allow_pickle=True)[()]
model = model['9Cr-001']

#C data
C_data = np.load(path + '/constant_matcher_score_lib.npy', 
        allow_pickle=True)[()]
C_data = {k: v['C'] for k, v in C_data.items()}

#Load the 9Cr data
ID = [1, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
      43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60,
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
      77, 78, 79, 82]
ID = ['9Cr-{}'.format(str(i).zfill(3)) for i in ID]

path = '/Users/mamu867/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data'
df = pd.read_csv(path + '/Cleaned_data.csv')
df = df[df.ID.isin(ID)]

ele = ['Fe', 'C', 'Cr', 'Mn', 'Si', 'Ni', 'Co', 'Mo', 'W', 'Nb', 'Al',
       'P', 'Cu', 'Ti', 'Ta', 'Hf', 'Re', 'V', 'B', 'N', 'O', 'S', 'Zr']
df[ele] = df[ele].fillna(0)

df['log_CT_CS'] = np.log(df['CT_CS'])

df['LMP_Model'] = df.apply(lambda x:
        1e-3 * x['CT_Temp'] * (np.log(x['CT_RT']) + C_data[x['ID']]), axis=1)

features = [i for i in df.columns if i not in ['CT_RT', 'CT_Temp', 
    'ID', 'CT_CS', 'LMP_Model']]
X = df[features].to_numpy(np.float32)
y = df['LMP_Model'].to_numpy(np.float32)

pd = ProcessData(X=X, y=y, features=features)
pd.clean_data()
data = pd.get_data()
del pd

'''
parameters_grid = {'boosting_type': ['gbdt', 'goss'],
              'num_leaves': [100, 200],
              'max_depth': [-1],
              'learning_rate': [0.01],
              'n_estimators': [100, 200],
              'subsample_for_bin': [200000],
              'objective': [None],
              'class_weight': [None],
              'min_split_gain': [0.0],
              'min_child_weight': [0.001],
              'min_child_samples': [20],
              'subsample': [1.0],
              'subsample_freq': [0],
              'colsample_bytree': [1.0],
              'reg_alpha': [0.0],
              'reg_lambda': [0.0],
              'random_state': [42],
              'n_jobs': [-1],
              'silent': [True],
              'importance_type' : ['split'],
              'num_boost_round': [2000],
              'tree_learner': ['serial', 'feature', 'data', 'voting'],
              'boost_from_average': [True, False],
              'alpha': [0.1, 0.5, 0.9, 1.0]}
              #'bagging_fraction': [0.2]}

lgb = LGBM(X=X,
           y=y,
           cv=5,
           grid_search=True,
           eval_metric='rmse',
           param_grid=parameters_grid)

lgb.run_model()
print(lgb.__dict__)
'''

parameters = {'boosting_type': 'dart',
              'num_leaves': 1000,
              'max_depth': -1,
              'learning_rate': 0.8,
              'n_estimators': 50,
              'subsample_for_bin': 200000,
              'objective': None,
              'class_weight': None,
              'min_split_gain': 0.0,
              'min_child_weight': 0.001,
              'min_child_samples': 20,
              'subsample': 1.0,
              'subsample_freq': 0,
              'colsample_bytree': 1.0,
              'reg_alpha': 0.0,
              'reg_lambda': 0.0,
              'random_state': 42,
              'n_jobs':-1,
              'silent':True,
              'importance_type' :'split',
              'num_boost_round': 100,
              'tree_learner': 'feature'}

lgb = GBM(package='lightgbm',
          X=data['X'],
          y=data['y'],
          cv=5,
          grid_search=False,
          eval_metric='rmse',
          parameters=parameters)

lgb.run_model()
print(lgb.__dict__)
print(len(y))



