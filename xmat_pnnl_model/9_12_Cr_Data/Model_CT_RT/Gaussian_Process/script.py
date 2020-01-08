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

#Load the features
df = pd.read_csv('../../9_12_Cr.csv')
df = df.dropna(how='all')
df = df.dropna(subset=['RT', 'CS', 'CT Temp'])
df['log_CS'] = np.log(df['CS'])
X = df[[i for i in df.columns if i not in ['ID', 'RT', 
    'CS']]].to_numpy(dtype=np.float)
y = df['RT'].to_numpy()

pd = ProcessData(X=X, y=y, metadata=[i for i in df.columns 
    if i not in ['ID', 'RT', 'CS']])
pd.clean_data()
X, y, metadata = pd.get_data()
del pd

kernel = 1.0*RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0) 
kernel += 1.0*DotProduct(sigma_0=1.0) + 1.0*Matern(length_scale=1.0)
skgp = SKGP(X=X, y=y, kernel=kernel, validation='5-Fold')
skgp.run_GP()
skgp.__dict__['features'] = metadata
print(skgp.__dict__)

np.save('gp_run.npy', skgp.__dict__)
skgp.plot_parity(data='train', err_bar=True).savefig(
        'train_parity_plot.png')
skgp.plot_parity(data='test', err_bar=True).savefig(
        'test_parity_plot.png')
