#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: script.py
#AUTHOR: Osman Mamun
#DATE CREATED: 12-10-2019

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.simplefilter("ignore", UserWarning)
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
import sys
from scipy.optimize import fmin

from xmat_pnnl_code import PolyFit
from xmat_pnnl_code import Shapley

def get_C(A):
    C0 = 25
    C = fmin(compute_score, C0, ftol=1e-4, 
            args=(A, base_model), full_output=1)
    return C[0][0] - C[1]

def compute_score(C, *args):
    df = args[0]
    model = args[1]
    df['LMP'] = 1e-3 * (df['CT_Temp']) * (np.log(df['CT_RT']) + C)
    X = df[['LMP']].to_numpy()
    return -model.score(X, np.log(df['CT_CS']))

ID = [1, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
      43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60,
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
      77, 78, 79, 80, 82]
ID = ['9Cr-{}'.format(str(i).zfill(3)) for i in ID]

data = pd.read_csv('../../Cleaned_data.csv')
keep_column = ['ID', 'CT_Temp', 'CT_CS', 'CT_RT']
data = data[keep_column]
C = 25
data['LMP'] = 1e-3 * (data['CT_Temp']) * (np.log(data['CT_RT']) + C)
def d_list(): return defaultdict(list)
score_lib = defaultdict(d_list)
slope_lib = defaultdict(d_list)
intercept_lib= defaultdict(d_list)
C_lib= defaultdict(d_list)

base_model = PolyFit(data[data['ID'] == '9Cr-001'][['LMP', 'CT_CS']],
                     target='CT_CS',
                     degree=1,
                     logify_y=True)
base_model.fit()
base_model = base_model.model

for alloy_id in ID:
    print('Working on alloy: {}'.format(alloy_id))
    df = data[data['ID'] == alloy_id]
    n_samp = len(df)
    for comb in np.arange(2, n_samp+1):
        for _ in range(10):
            rows = np.random.choice(np.arange(len(df)), comb)
            poly = PolyFit(df=df[['LMP', 'CT_CS']].iloc[rows], 
                            target='CT_CS', 
                            degree=1,
                            logify_y=True)
            poly.fit()
            C_lib[alloy_id][comb].append(get_C(df.iloc[rows]))
            score_lib[alloy_id][comb].append(poly.score)
            slope_lib[alloy_id][comb].append(poly.model.coef_[0])
            intercept_lib[alloy_id][comb].append(poly.model.intercept_)
    
    y = np.arange(2, n_samp+1)
    score = [np.mean(score_lib[alloy_id][i]) for i in y]
    score_std = [np.std(score_lib[alloy_id][i]) for i in y]
    slope = [np.mean(slope_lib[alloy_id][i]) for i in y]
    slope_std = [np.std(slope_lib[alloy_id][i]) for i in y]
    intercept = [np.mean(intercept_lib[alloy_id][i]) for i in y]
    intercept_std = [np.std(intercept_lib[alloy_id][i]) for i in y]
    C = [np.mean(C_lib[alloy_id][i]) for i in y]
    C_std = [np.std(C_lib[alloy_id][i]) for i in y]
    
    plt.grid(color='grey', linewidth=0.5, alpha=0.3, linestyle='--')
    plt.plot(y, score, '-', linewidth=4, c='#1A4876', label='score')
    plt.errorbar(y, score, yerr=score_std, fmt='.k', color='black', alpha=0.3)
    plt.xlabel('# data points')
    plt.ylabel('correlation coefficient')
    plt.savefig('learning_curves/{}_score_plot.png'.format(alloy_id))
    plt.clf()
    plt.grid(color='grey', linewidth=0.5, alpha=0.3, linestyle='--')
    plt.plot(y, slope, '-', linewidth=4, c='#6DAE81', label='slope')
    plt.errorbar(y, slope, yerr=slope_std, fmt='.k', color='black', alpha=0.3)
    plt.xlabel('# data points')
    plt.ylabel('slope')
    plt.savefig('learning_curves/{}_slope_plot.png'.format(alloy_id))
    plt.clf()
    plt.grid(color='grey', linewidth=0.5, alpha=0.3, linestyle='--')
    plt.plot(y, intercept, '-', linewidth=4, c='#158078', label='intercept')
    plt.errorbar(y, intercept, yerr=intercept_std, fmt='.k', color='black', alpha=0.3)
    plt.xlabel('# data points')
    plt.ylabel('intercept')
    plt.savefig('learning_curves/{}_intercept_plot.png'.format(alloy_id))
    plt.clf()
    plt.grid(color='grey', linewidth=0.5, alpha=0.3, linestyle='--')
    plt.plot(y, C, '-', linewidth=4, c='#87A96B', label='intercept')
    plt.errorbar(y, C, yerr=C_std, fmt='.k', color='black', alpha=0.3)
    plt.xlabel('# data points')
    plt.ylabel('C')
    plt.savefig('learning_curves/{}_C_plot.png'.format(alloy_id))
    plt.clf()
    
np.save('score_lib.npy', score_lib)
np.save('slope_lib.npy', slope_lib)
np.save('intercept_lib.npy', intercept_lib)
np.save('C_lib.npy', C_lib)

