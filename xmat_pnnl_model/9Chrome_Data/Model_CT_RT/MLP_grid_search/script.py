#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: script.py
#AUTHOR: Osman Mamun
#DATE CREATED: 12-17-2019

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from xmat_pnnl_code import ProcessData
from xmat_pnnl_code import SKGridReg
from xmat_pnnl_code import SKREG

#Load the data
path = '/Users/mamu867/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data/LMP/'
data = np.load(path + 'constant_matcher_score_lib.npy', allow_pickle=True)[()]
del data['9Cr-080'] #questionable action

#Load the features
path = '/Users/mamu867/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data/'
path += 'Generate_New_Fingerprints/fingerprints_data.csv'
df = pd.read_csv(path)
df.replace('ND', np.nan, inplace=True)
ID = [1, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
      43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60,
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
      77, 78, 79, 82]
ID = ['9Cr-{}'.format(str(i).zfill(3)) for i in ID]
df = df[df.ID.isin(ID)]
df['log_CT_CS'] = np.log(df['CT_CS'])

#default values to fill in missiing values for the following features
fillvals = {'Fe': 0,'C':0,'Cr':0,'Mn':0,
            'Si':0,'Ni':0,'Co':0,'Mo':0,
            'W':0,'Nb':0,'Al':0,'P':0,
            'Cu':0,'Ti':0,'Ta':0,'Hf':0,
            'Re':0,'V':0,'B':0,'N':0,
            'O':0,'S':0,'Homo':0,'Normal':25,
            'Temper1':25,'Temper2':25,'Temper3':25}

#Drop any rows where a missing value in the following features exist
dropna_cols = ['ID', 'CT_Temp', 'log_CT_CS', 'CT_RT', 'AGS No.']

#Features/Columns to remove from dataset
exclude_cols = ['CT_0.1% CS', 'CT_0.2% CS', 'CT_0.5% CS',
                'CT_1.0% CS', 'CT_2.0% CS', 'CT_5.0% CS',
                'CT_EL', 'CT_RA', 'CT_MCR', 'CT_0.15%CS', 
                'CT_0.25% CS', 'CT_TTC', 'Homo']
df = df.drop(exclude_cols, axis=1)
df = df.dropna(how='all')
#Drop all rows with missing data from columns in dropna_cols list
df = df.dropna(subset=dropna_cols)
#Replace missing values for columns with corresponding fill values
df = df.fillna(value=fillvals)

W_col = [i for i in df.columns if i.startswith('Weighted_')]
X = df[[i for i in df.columns if i not in ['ID', 'CT_RT', 
    'CT_CS'] + W_col]].to_numpy(dtype=np.float)
y = df['CT_RT'].to_numpy()
pd = ProcessData(X=X, y=y, metadata=[i for i in df.columns
    if i not in ['ID', 'CT_RT', 'CT_CS']])
pd.impute_data()
pd.scale_data()
X, y, metadata = pd.get_data()
del pd

param_space = {'hidden_layer_sizes': [(100,), (100, 100,), (100, 100, 100,)],
               'max_iter': [250, 300, 345, 400],
               'activation': ['relu'],
               'solver': ['lbfgs'],
               'alpha': [0.01, 0.02, 0.03, 0.04],
               'learning_rate': ['constant', 'adaptive']}


mlpgrid = SKGridReg(X=X,
                    y=y,
                    estimator='MLP',
                    estimator_param_space=param_space)
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

