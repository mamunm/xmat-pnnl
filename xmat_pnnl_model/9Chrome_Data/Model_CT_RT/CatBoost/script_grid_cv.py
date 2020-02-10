#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: script.py
#AUTHOR: Osman Mamun
#DATE CREATED: 12-19-2019

import numpy as np
import sys
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xmat_pnnl_code import ProcessData
from xmat_pnnl_code import GBM
from lightgbm import plot_importance, plot_metric, plot_tree
import matplotlib.pyplot as plt
import json

#Model data
path = '/Users/mamu867/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data/LMP'
model = np.load(path + '/model_params.npy', allow_pickle=True)[()]
model = model['9Cr-001']

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
df = df.dropna(subset=['CT_RT', 'CT_CS', 'CT_EL', 'CT_RA', 'CT_Temp',
    'Normal', 'Temper1', 'AGS No.', 'CT_MCR'])
df['log_CT_CS'] = np.log(df['CT_CS'])

features = [i for i in df.columns if i not in ['CT_RT', 'CT_CS', 'ID']]
X = df[features].to_numpy(np.float32)
y = df['CT_RT'].to_numpy(np.float32)

pdata = ProcessData(X=X, y=y, features=features)
pdata.clean_data()
data = pdata.get_data()
del pdata

param_grid = {'iterations': [500, 1000, 2000, 5000, 10000],
              #'learning_rate': [None],
              #'depth': [4, 6, 8, 10, 12],
              #'l2_leaf_reg': [None],
              #'model_size_reg': [None],
              #'rsm': [None],
              'loss_function': ['RMSE'],
              #'border_count': [None],
              #'feature_border_type': [None],
              #'per_float_feature_quantization': [None],
              #'input_borders': [None],
              #'output_borders': [None],
              #'fold_permutation_block': [None],
              #'od_pval': [None],
              #'od_wait': [None],
              #'od_type': [None],
              #'nan_mode': [None],
              #'counter_calc_method': [None],
              #'leaf_estimation_iterations': [None],
              #'leaf_estimation_method': [None],
              #'thread_count': [None],
              #'random_seed': [None],
              #'use_best_model': [None],
              #'best_model_min_trees': [None],
              #'custom_metric': [None],
              #'eval_metric': [None],
              'bagging_temperature': [0, 0.5, 1, 20],
              'boosting_type': ['Ordered', 'Plain'],
              'bootstrap_type': ['Bayesian', 'Bernoulli', 'MVS', 'Poisson'],
              #'subsample': [None],
              'max_depth': [6, 10, 16],
              #'n_estimators': [20, 50, 100, 500],
              #'num_boost_round': [None],
              #'num_trees': [None],
              #'reg_lambda': [None],
              #'objective': [None],
              #'eta': [None],
              #'early_stopping_rounds': [None],
              #'cat_features': [None],
              'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide'],
              #'min_data_in_leaf': [None],
              #'min_child_samples': [None],
              #'max_leaves': [None],
              #'num_leaves': [None],
              #'score_function': [None],
              #'leaf_estimation_backtracking': [None],
              #'ctr_history_unit': [None],
              #'monotone_constraints': [None]}
              }

catboost = GBM(package='catboost',
          X=data['X'],
          y=data['y'],
          feature_names=data['features'],
          cv=10,
          grid_search=True,
          grid_search_scoring='r2',
          param_grid=param_grid,
          eval_metric='rmse')


catboost.run_model()
print(catboost.__dict__)
np.save('catboost_grid_res.npy', catboost.__dict__)
