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
from xmat_pnnl_code import SKREG
from xmat_pnnl_code import SKGridReg

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
df['log_CT_CS'] = np.log(df['CT_CS'])

df['LMP_Model'] = df.apply(lambda x:
        1e-3 * x['CT_Temp'] * (np.log(x['CT_RT']) + 25), axis=1)

features = [i for i in df.columns if i not in ['CT_RT', 'CT_Temp', 
    'ID', 'CT_CS', 'LMP_Model']]
X = df[features].to_numpy(np.float32)
y = df['LMP_Model'].to_numpy(np.float32)

pd = ProcessData(X=X, y=y, metadata=features)
pd.clean_data()
X, y, metadata = pd.get_data()
del pd

param_space = {'max_iter': [5000],
               'activation': ['relu'],
               'solver': ['lbfgs'],
               'alpha':[0.001, 0.002, 0.003, 0.004, 0.005, 0.006,
                        0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
               'learning_rate': ['constant']}


mlpgrid = SKGridReg(X=X,
                    y=y,
                    estimator='MLP',
                    estimator_param_space=param_space,
                    cv=10)
mlpgrid.run_grid_search()
print(mlpgrid.__dict__)
np.save('grid_results_without_weighted.npy', mlpgrid.__dict__)

skmlp = SKREG(X=X,
              y=y,
              estimator='MLP',
              estimator_param = mlpgrid.best_params,
              validation='5-Fold')
skmlp.run_reg()
skmlp.__dict__['features'] = metadata
print(skmlp.__dict__)

np.save('mlp_run.npy', skmlp.__dict__)
skmlp.plot_parity(data='train').savefig('train_parity_plot.png')
skmlp.plot_parity(data='test').savefig('test_parity_plot.png')
