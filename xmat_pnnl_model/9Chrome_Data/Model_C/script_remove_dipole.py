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

#Load the data
path = '/Users/mamu867/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data/LMP/'
data = np.load(path + 'constant_matcher_score_lib.npy', allow_pickle=True)[()]
del data['9Cr-080'] #questionable action

#Load the features
path = '/Users/mamu867/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data/'
path += 'Generate_New_Fingerprints/fingerprints_data.csv'
df = pd.read_csv(path)

agg_data = []

f_col = [i for i in df.columns if i.startswith('Weighted')]

for k, v in data.items():
    SRS = df[df.ID == k].iloc[0]
    f = [SRS[i] for i in f_col]
    agg_data.append([k] + f + [v['C']])

columns=['ID'] + f_col + ['C']

df = pd.DataFrame(agg_data, columns=columns)

'''
#Explore individual model
features = ['Atomic Number', 'Melting Point', 'Metallic Radius']
for fp in features:
    lr = LinFit(df=df[[fp, 'C']], target='C', prop=fp)
    lr.fit()
    print('Score for {}: {}'.format(fp, lr.score))
    lr.plot_model().show()
    plt.clf()
    lr.plot_parity().show()
    plt.clf()

g = sns.PairGrid(df, x_vars=[i for i in df.columns if i not in['ID', 'C']], 
        y_vars='C')    
g = g.map(plt.scatter)
plt.show()
'''

'''
for i in df.columns:
    if i in ['ID', 'C']:
        continue
    df.plot(x=i, y='C', kind='scatter')
    plt.savefig('scatter_plot/{}.png'.format(i))
    plt.clf()
'''

X = df[[i for i in df.columns 
    if i not in ['ID', 'C', 'Weighted_dipole_polarizability']]].to_numpy()
y = df['C'].to_numpy()

pd = ProcessData(X=X, y=y, metadata=[i for i in df.columns 
    if i not in ['ID', 'C']])
pd.clean_data()
X, y, metadata = pd.get_data()
del pd

kernel = 1.0*RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0) 
kernel += 1.0*DotProduct(sigma_0=1.0) + 1.0*Matern(length_scale=1.0)
skgp = SKGP(X=X, y=y, kernel=kernel, validation='leave_one_out')
skgp.run_GP()
skgp.__dict__['features'] = metadata
print(skgp.__dict__)

np.save('gp_run_remove_dipole.npy', skgp.__dict__)
skgp.plot_parity(data='train', err_bar=True).savefig(
        'train_parity_plot_remove_dipole.png')
skgp.plot_parity(data='test', err_bar=True).savefig(
        'test_parity_plot_remove_dipole.png')
