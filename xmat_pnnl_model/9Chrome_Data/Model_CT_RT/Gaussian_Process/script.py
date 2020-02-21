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
from xmat_pnnl_code import SKGP
from sklearn.gaussian_process.kernels import (RBF, WhiteKernel,
        DotProduct, Matern)
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

features = [i for i in df.columns if i not in ['CT_RT', 'CT_CS',
    'CT_MCR', 'ID']]
X = df[features].to_numpy(np.float32)
y = df['CT_RT'].to_numpy(np.float32)

pdata = ProcessData(X=X, y=y, features=features)
#pdata.clean_data(scale_strategy={'strategy': 'power_transform',
#    'method': 'yeo-johnson'})
pdata.clean_data(scale_strategy={'strategy': 'StandardScaler'})
data = pdata.get_data()
scale = pdata.scale
del pdata

kernel = 1.0*RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0) 
kernel += 1.0*DotProduct(sigma_0=1.0) + 1.0*Matern(length_scale=1.0)
skgp = SKGP(X=data['X'], y=data['y'], 
        kernel=kernel, validation='5-Fold')
skgp.run_GP()
skgp.__dict__['features'] = data['features']
print(skgp.__dict__)

np.save('gp_run.npy', skgp.__dict__)
skgp.plot_parity(data='train', err_bar=True).savefig(
        'train_parity_plot.png')
skgp.plot_parity(data='test', err_bar=True).savefig(
        'test_parity_plot.png')
