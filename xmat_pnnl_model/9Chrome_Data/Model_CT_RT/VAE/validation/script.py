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
from xmat_pnnl_code import GBM, AutoEncoder
from lightgbm import plot_importance, plot_metric, plot_tree
import matplotlib.pyplot as plt
import json
import xmat_pnnl_code as xcode
import shap
from scipy.stats import pearsonr as pr
from sklearn.decomposition import PCA

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
features = [i for i in features if 'Weighted' not in i]
X = df[features].to_numpy(np.float32)
y = df['CT_RT'].to_numpy(np.float32)

pdata = ProcessData(X=X, y=y, features=features)
#pdata.clean_data(scale_strategy={'strategy': 'power_transform', 
#    'method': 'yeo-johnson'})
pdata.clean_data()
data = pdata.get_data()
scale = pdata.scale
del pdata

ae = AutoEncoder(arch=[data['X'].shape[1], 12, 6, 2],
        X=data['X'], loss='xent', epochs=1500)

ae.build_model()
ae.save_model("9Chrome_vae")
X_reconstructed = ae.vae.predict(data['X'])
pca_reconstructed = PCA(n_components=2).fit_transform(X_reconstructed)
pca_real = PCA(n_components=2).fit_transform(data['X'])

plt.figure(figsize=(6, 6))
plt.title('Principal Component Analysis for VAE validation.', fontsize=16)
plt.grid(color='blue', linestyle='dotted', linewidth=0.8)
plt.xlabel(r"$X_1$", fontsize=16)
plt.ylabel(r"$X_2$", fontsize=16)
plt.scatter(pca_reconstructed[:, 0], pca_reconstructed[:, 1],
            marker='h', s=50, color='darkred', alpha=0.8,
            label='Reconstructed samples')
plt.scatter(pca_real[:, 0], pca_real[:, 1], marker='h', s=50,
            color='darkblue', alpha=0.5, label='Real samples')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best', fontsize=14)
plt.show()
np.save("validation_data.npy", {"X_real": data["X"], 
    "X_reconstructed": X_reconstructed})




