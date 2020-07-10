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
ae.save_model("aus_vae")
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




