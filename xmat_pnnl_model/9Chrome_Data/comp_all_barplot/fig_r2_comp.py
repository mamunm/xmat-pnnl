#!/usr/bin/env python
# -*-coding: utf-8 -*-

#script.py
#Osman Mamun
#DATE CREATED: 10-22-2018

import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['axes.axisbelow'] = True

scheme = ['CT_RT', 'LMP(C=25)', 'LMP (Adjusted C)']
r2_catboost_train = [0.96, 0.98, 0.984]
r2_catboost_train_std = [0.03, 0.001, 0.002]
r2_xgboost_train = [0.99, 0.998, 0.998]
r2_xgboost_train_std = [0.008, 0.001, 0.0002]
r2_lightgbm_train = [0.92, 0.986, 0.99]
r2_lightgbm_train_std = [0.07, 0.004, 0.002]
r2_catboost_test = [0.84, 0.85, 0.82]
r2_catboost_test_std = [0.10, 0.09, 0.11]
r2_xgboost_test = [0.82, 0.82, 0.82]
r2_xgboost_test_std = [0.08, 0.08, 0.09]
r2_lightgbm_test = [0.70, 0.75, 0.79]
r2_lightgbm_test_std = [0.07, 0.11, 0.11]

plt.figure(facecolor='white', figsize=(30, 10))
#plt.style.use('seaborn-white')

index = np.arange(len(scheme))
bar_width = 0.15
opacity = 1

plt.grid(c='grey', linestyle='--', alpha=0.2, axis='y')


rects1 = plt.bar(index, r2_catboost_train, bar_width,
                 alpha=opacity,
                 yerr=r2_catboost_train_std,
                 color='#1A4876',
                 label='CatBoost Train')

rects2 = plt.bar(index+bar_width, r2_catboost_test, bar_width,
                 alpha=opacity,
                 yerr=r2_catboost_train_std,
                 color='#7366BD',
                 label='CatBoost Test')

rects3 = plt.bar(index+2*bar_width, r2_lightgbm_train, bar_width,
                 alpha=opacity,
                 yerr=r2_lightgbm_train_std,
                 color='#8E4585',
                 label='LightGBM Train')

rects4 = plt.bar(index+3*bar_width, r2_lightgbm_test, bar_width,
                 alpha=opacity,
                 yerr=r2_lightgbm_test_std,
                 color='#158078',
                 label='LightGBM Test')

rects5 = plt.bar(index+4*bar_width, r2_xgboost_train, bar_width,
                 alpha=opacity,
                 yerr=r2_xgboost_train_std,
                 color='#6DAE81',
                 label='XGBoost Train')

rects6 = plt.bar(index+5*bar_width, r2_xgboost_test, bar_width,
                 alpha=opacity,
                 yerr=r2_xgboost_test_std,
                 color='#7366BD',
                 label='XGBoost Test')

#plt.axhline(y=0.10, linewidth=3, color='red', linestyle='-')
plt.xlabel('Scheme', fontsize=30)
plt.ylabel('Correlation Coefficient', fontsize=30)
plt.xticks(index + 2.5 * bar_width, scheme,
           fontsize=30) # rotation=45)
plt.yticks(fontsize=30)
plt.legend(loc='upper right', fontsize=15, framealpha=0.2, fancybox=True)
#plt.tight_layout()
plt.savefig('fig_r2_CT_RT.png')
plt.show()

