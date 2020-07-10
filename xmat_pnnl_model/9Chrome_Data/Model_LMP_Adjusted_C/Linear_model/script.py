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

def get_LMP(A):
    return (A['log_CT_CS'] - model['intercept']) / model['coef']

#Model data
path = '/Users/osmanmamun/PNNL_Mac/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data/LMP'
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

path = '/Users/osmanmamun/PNNL_Mac/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data'
df = pd.read_csv(path + '/Cleaned_data.csv')
df = df[df.ID.isin(ID)]
df = df[['ID', 'CT_CS', 'CT_Temp', 'CT_RT']]
df['log_CT_CS'] = np.log(df['CT_CS'])

#LMP should be x=(y-c)/m
df['LMP'] = df.apply(lambda x: 
        (x['log_CT_CS']-model['intercept'])/model['coef'][0], axis=1)

df['LMP_Model'] = df.apply(lambda x:
        1e-3 * x['CT_Temp'] * (np.log(x['CT_RT']) + C_data[x['ID']]), axis=1)
df['LMP_Model_2'] = df.apply(lambda x:
        1e-3 * x['CT_Temp'] * (np.log(x['CT_RT']) + 25), axis=1)
df['Prediction'] = df.apply(lambda x: 
        np.exp((1e3*x['LMP']/x['CT_Temp'])-C_data[x['ID']]), axis=1)

df['C'] = df.apply(lambda x: C_data[x['ID']], axis=1)
df['diff'] = df['CT_RT'] - df['Prediction']
print(df)
#df.plot(x='LMP_Model', y='log_CT_CS', kind='scatter')
#plt.show()
#df.plot(x='LMP_Model_2', y='log_CT_CS', color='r', kind='scatter')
#plt.show()
plt.scatter(df['LMP_Model'], df['log_CT_CS'])
plt.plot(df['LMP'], df['log_CT_CS'], c='r', linewidth=4)
plt.savefig('LMP_log_CT_CS.png')
df.plot(x='CT_RT', y='Prediction', kind='scatter')
plt.savefig('CT_RT_Prediction.png')
df.to_csv('final_df.csv')

lr = LinearRegression().fit(df['LMP_Model'].to_numpy().reshape(-1, 1), 
        df['log_CT_CS'].to_numpy())
print('R2 value for the model: {}'.format(linregress(
    lr.predict(df['LMP_Model'].to_numpy().reshape(-1, 1)), 
    df['log_CT_CS'])[2]**2))

print('R2 value for the prediction: {}'.format(linregress(
    df['CT_RT'], df['Prediction'])[2]**2))
print(model)
