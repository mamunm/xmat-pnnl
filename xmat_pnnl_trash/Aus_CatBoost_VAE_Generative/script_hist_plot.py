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
X = df[features].to_numpy(np.float32)
y = df['CT_RT'].to_numpy(np.float32)
pdata = ProcessData(X=X, y=y, features=features)
pdata.clean_data(scale_strategy={'strategy': 'StandardScaler'})
data = pdata.get_data()
scale = pdata.scale
del pdata

ae = AutoEncoder([data['X'].shape[1], 20, 16, 12, 6], X=data['X'],
                loss='mse', epochs=250)
ae.build_model()
hist = ae.get_hist()
hist.plot()
plt.title('Austenitic steel dataset', fontsize=18)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('MSE', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('training_VAE.png', dpi=600)
plt.show()

