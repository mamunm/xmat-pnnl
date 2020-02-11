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
from xmat_pnnl_code import ProcessData
from xmat_pnnl_code import GBM
import shap
import xmat_pnnl_code as xcode

#Model data
base_path = '/'.join(xcode.__path__[0].split('/')[:-1])
path = base_path + '/data_processing/9Cr_data/LMP'
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

df['LMP_Model'] = df.apply(lambda x:
        1e-3 * x['CT_Temp'] * (np.log(x['CT_RT']) + C_data[x['ID']]), axis=1)

features = [i for i in df.columns if i not in ['CT_RT', 'CT_Temp', 
    'ID', 'CT_CS', 'LMP_Model', 'CT_MCR']]
X = df[features].to_numpy(np.float32)
y = df['LMP_Model'].to_numpy(np.float32)
y2 = df[['ID', 'CT_RT', 'CT_Temp', 'CT_CS']].values.tolist()
pdata = ProcessData(X=X, y=y, y2=y2, features=features)
pdata.clean_data()
data = pdata.get_data()
scale = pdata.scale
del pdata

parameters = {'booster': 'gbtree', #gbtree, gblinear, dart
              'eta': 0.3, #learning rate
              'gamma': 0, #min split loss
              'max_depth': 10,
              'tree_method': 'auto'}

CT_RT = np.array([i[1] for i in data['y2']])
CT_Temp = np.array([i[2] for i in data['y2']])
CT_CS = np.array([i[3] for i in data['y2']])
ID = [i[0] for i in data['y2']]
C = np.array([C_data[i] for i in ID])

xgb = GBM(package='xgboost',
          X=data['X'],
          y=data['y'],
          cv=5,
          grid_search=False,
          eval_metric='rmse',
          parameters=parameters,
          CT_Temp=CT_Temp,
          CT_RT=CT_RT,
          C=C)

xgb.run_model()
print(xgb.__dict__)
xgb.parity_plot(data='train', quantity='LMP').savefig('parity_LMP_train.png')
xgb.parity_plot(data='test', quantity='LMP').savefig('parity_LMP_test.png')
xgb.parity_plot(data='train', quantity='CT_RT').savefig('parity_CT_RT_train.png')
xgb.parity_plot(data='test', quantity='CT_RT').savefig('parity_CT_RT_test.png')
np.save('xgb_dict.npy', xgb.__dict__)
plt.clf()
explainer = shap.TreeExplainer(xgb.model[-1])
shap_values = explainer.shap_values(data['X'])

XX = scale.inverse_transform(data['X'])
X = pd.DataFrame(XX, columns=data['features'])
# summarize the effects of all the features
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
