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
import shap 
import xmat_pnnl_code as xcode

#Model data
base_path = '/'.join(xcode.__path__[0].split('/')[:-1])
path = base_path + '/data_processing/9Cr_data/LMP'
model = np.load(path + '/model_params.npy', allow_pickle=True)[()]
model = model['9Cr-001']

#Load the 9Cr data
ID = [1, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
      43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60,
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
      77, 78, 79, 82]
ID = ['9Cr-{}'.format(str(i).zfill(3)) for i in ID]

path = base_path + '/data_processing/9Cr_data'
df = pd.read_csv(path + '/Cleaned_data.csv')
df = df[df.ID.isin(ID)]
ele = ['Fe', 'C', 'Cr', 'Mn', 'Si', 'Ni', 'Co', 'Mo', 'W', 'Nb', 'Al',
       'P', 'Cu', 'Ti', 'Ta', 'Hf', 'Re', 'V', 'B', 'N', 'O', 'S', 'Zr']
df[ele] = df[ele].fillna(0)
df = df.dropna(subset=['CT_RT', 'CT_CS', 'CT_EL', 'CT_RA', 'CT_Temp',
    'Normal', 'Temper1', 'AGS No.', 'CT_MCR'])
df['log_CT_CS'] = np.log(df['CT_CS'])
df['log_CT_MCR'] = np.log(df['CT_MCR'])

df['LMP_Model'] = df.apply(lambda x:
        1e-3 * x['CT_Temp'] * (np.log(x['CT_RT']) + 25), axis=1)

features = [i for i in df.columns if i not in ['CT_RT', 'CT_Temp', 
    'ID', 'CT_CS', 'LMP_Model', 'CT_MCR']]
X = df[features].to_numpy(np.float32)
y = df['LMP_Model'].to_numpy(np.float32)
y2 = df[['ID', 'CT_RT', 'CT_Temp', 'CT_CS']].values.tolist()

pdata = ProcessData(X=X, y=y, y2=y2, features=features)
pdata.clean_data()
data = pdata.get_data()
scale = pdata.scale
del pdata

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

CT_RT = np.array([i[1] for i in data['y2']])
CT_Temp = np.array([i[2] for i in data['y2']])
ID = [i[0] for i in data['y2']]
C = np.array([25 for i in ID])

catboost = GBM(package='catboost',
          X=data['X'],
          y=data['y'],
          feature_names=data['features'],
          cv=5,
          grid_search=False,
          eval_metric='rmse',
          parameters=parameters,
          CT_Temp=CT_Temp,
          CT_RT=CT_RT,
          C=C)


catboost.run_model()
print(catboost.__dict__)
catboost.parity_plot(data='train', quantity='LMP').savefig('parity_LMP_train.png')
catboost.parity_plot(data='test', quantity='LMP').savefig('parity_LMP_test.png')
catboost.parity_plot(data='train', quantity='CT_RT').savefig('parity_CT_RT_train.png')
catboost.parity_plot(data='test', quantity='CT_RT').savefig('parity_CT_RT_test.png')
np.save('xgb_dict.npy', catboost.__dict__)
plt.clf()
explainer = shap.TreeExplainer(catboost.model[-1])
shap_values = explainer.shap_values(data['X'])

XX = scale.inverse_transform(data['X'])
X = pd.DataFrame(XX, columns=data['features'])
# summarize the effects of all the features
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
