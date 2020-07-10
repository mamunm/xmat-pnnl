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
from xmat_pnnl_code import VaeGBM
from xmat_pnnl_code import AutoEncoder
from lightgbm import plot_importance, plot_metric, plot_tree
import matplotlib.pyplot as plt
import json
import xmat_pnnl_code as xcode
import shap
from keras.utils import plot_model

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
features.append('CT_RT')
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
              'num_boost_round': 50000,
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

catboost = VaeGBM(package='catboost',
          X=data['X'],
          scale=scale,
          features=data['features'],
          element = [i for i in ele if i in data['features']],
          cv=3,
          gen_n=400,
          eval_metric='rmse',
          parameters=parameters)


catboost.run_model()
print(catboost.__dict__)
np.save('catboost_res.npy', catboost.__dict__)
'''
catboost.parity_plot(data='train', 
        quantity='CT_RT', scheme=1).savefig('parity_CT_RT_train.png')
catboost.parity_plot(data='test', 
        quantity='CT_RT', scheme=1).savefig('parity_CT_RT_test.png')
plt.clf()
explainer = shap.TreeExplainer(catboost.model[-1])
shap_values = explainer.shap_values(data['X'])

XX = scale.inverse_transform(data['X'])
X = pd.DataFrame(XX, columns=data['features'])
# summarize the effects of all the features
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
'''
