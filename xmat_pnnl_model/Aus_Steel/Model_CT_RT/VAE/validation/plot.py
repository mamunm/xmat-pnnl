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

data = np.load('validation_data.npy', allow_pickle=True)[()]

ae = AutoEncoder(arch=[data['X_real'].shape[1], 12, 6, 2],
        X=data['X_real'], loss='xent', epochs=1500, weights="aus_vae")

ae.build_model()
pca_reconstructed = PCA(n_components=2).fit_transform(data['X_reconstructed'])
pca_real = PCA(n_components=2).fit_transform(data['X_real'])

plt.figure(figsize=(9, 7))
#plt.title('Principal Component Analysis for VAE validation.', fontsize=16)
plt.grid(color='green', linestyle='dotted', linewidth=0.8)
plt.xlabel(r"$X_1$", fontsize=16)
plt.ylabel(r"$X_2$", fontsize=16)
plt.scatter(pca_reconstructed[:, 0], pca_reconstructed[:, 1],
            marker='h', s=50, color='darkred', alpha=0.8,
            label='Reconstructed samples')
plt.scatter(pca_real[:, 0], pca_real[:, 1], marker='h', s=50,
            color='darkblue', alpha=0.4, label='Real samples')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best', fontsize=14)
plt.savefig("pca_aus_validation.png")
plt.show()




