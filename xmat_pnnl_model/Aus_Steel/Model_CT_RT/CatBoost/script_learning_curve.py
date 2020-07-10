#!/usr/bin/env python
# -*-coding: utf-8 -*-

#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -J grid_search
#SBATCH --mail-user=mdosman.mamun@pnnl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

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
from xmat_pnnl_code import GBM_LC
from lightgbm import plot_importance, plot_metric, plot_tree
import matplotlib.pyplot as plt
import json
import xmat_pnnl_code as xcode
import shap

path = '/Users/osmanmamun/PNNL_Mac/PNNL_Code_Base/xmat-pnnl/data_processing/Aus_Steel_data'
df = pd.read_csv(path + '/Cleaned_data.csv')
ele = ["Fe", "C", "Cr", "Mn", "Si", "Ni", "Co", "Mo", "W", "Nb", "Al", "P",
        "Cu", "Ti", "V", "B", "N", "S"]
df[ele] = df[ele].fillna(0)
df = df.dropna(subset=['CT_RT', 'CT_CS', 'CT_EL', 'CT_RA', 'CT_Temp',
    'AGS No.'])
df['log_CT_CS'] = np.log(df['CT_CS'])
df['log_CT_MCR'] = np.log(df['CT_MCR'])
features = [i for i in df.columns if i not in ['CT_RT',
    'CT_CS', 'ID', 'CT_MCR']]
X = df[features].to_numpy(np.float32)
y = df['CT_RT'].to_numpy(np.float32)
pdata = ProcessData(X=X, y=y, features=features)
pdata.clean_data(scale_strategy={'strategy': 'StandardScaler'})
data = pdata.get_data()
scale = pdata.scale
del pdata

parameters = {'iterations': None,
              'learning_rate': None,
              'depth': None,
              'l2_leaf_reg': None,
              'model_size_reg': None,
              'rsm': None,
              'loss_function': 'RMSE',
              'border_count': None,
              'feature_border_type': None,
              'per_float_feature_quantization': None,
              'input_borders': None,
              'output_borders': None,
              'fold_permutation_block': None,
              'od_pval': None,
              'od_wait': None,
              'od_type': None,
              'nan_mode': None,
              'counter_calc_method': None,
              'leaf_estimation_iterations': None,
              'leaf_estimation_method': None,
              'thread_count': None,
              'random_seed': None,
              'use_best_model': None,
              'best_model_min_trees': None,
              'custom_metric': None,
              'eval_metric': None,
              'bagging_temperature': None,
              'boosting_type': None,
              'bootstrap_type': None,
              'subsample': None,
              'max_depth': None,
              'n_estimators': None,
              'num_boost_round': 5000,
              'num_trees': None,
              'reg_lambda': None,
              'objective': None,
              'eta': None,
              'early_stopping_rounds': None,
              'cat_features': None,
              'grow_policy': None,
              'min_data_in_leaf': None,
              'min_child_samples': None,
              'max_leaves': None,
              'num_leaves': None,
              'score_function': None,
              'leaf_estimation_backtracking': None,
              'ctr_history_unit': None,
              'monotone_constraints': None}


learning_dict = {i: None for i in [0.1, 0.2, 0.3, 0.4, 0.5, 
    0.6, 0.7, 0.8, 0.9]}
keys = ['N_dp', 'N_dp_train', 'N_dp_test', 'rmse_mean_train', 'rmse_std_train', 
        'rmse_mean_test', 'rmse_std_test', 'r2_mean_train', 'r2_std_train', 
        'r2_mean_test', 'r2_std_test', 'pr_mean_train', 'pr_std_train', 
        'pr_mean_test', 'pr_std_test']
for ts in learning_dict:
    catboost = GBM_LC(package='catboost',
              X=data['X'][data['y'] < 200000],
              y=data['y'][data['y'] < 200000],
              feature_names=data['features'],
              test_size=ts,
              eval_metric='rmse',
              parameters=parameters)

    catboost.run_model()
    learning_dict[ts] = dict([i for i in catboost.__dict__.items() 
        if i[0] in keys])
    print(learning_dict)
np.save('catboost_learning_dict.npy', learning_dict)
