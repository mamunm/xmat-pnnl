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
from bayesframe import BayesFrame
import json

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
X = df[[i for i in df.columns if i not in ['ID', 'CT_RT', 
    'CT_Temp', 'CT_CS']]].to_numpy(dtype=np.float)
df['LMP'] = df.apply(lambda x: 1e-3*(x['CT_Temp'])*(np.log(x['CT_RT']) 
    +data[x['ID']]['C']), axis=1)
y = df['LMP'].to_numpy()

dprocess = ProcessData(X=X, y=y, metadata=[i for i in df.columns 
    if i not in ['ID', 'CT_RT', 'CT_Temp', 'CT_CS']])
dprocess.clean_data()
X, y, metadata = dprocess.get_data()
del dprocess

df = pd.DataFrame(X, columns=metadata)
df['LMP'] = y

#Initialize the model
bframe = BayesFrame(df=df, target="LMP", val_scheme='5-Fold',
                    bic_scheme="per_n", model_scheme=["selection"])

#Print the best model
print(bframe.zoo)

np.save('best_model.npy', bframe.zoo)
