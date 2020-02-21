#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: parity_plot.py
#AUTHOR: Osman Mamun
#DATE CREATED: 02-18-2020

import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 

data = np.load('catboost_learning_dict.npy', allow_pickle=True)[()]
train_dp = np.array([data[i]['N_dp_train'] for i in data.keys()][::-1])
test_dp = np.array([data[i]['N_dp_test'] for i in data.keys()][::-1])
test_pcc = np.array([data[i]['pr_CT_RT_mean_test'] for i in data.keys()][::-1])
test_pcc_std = np.array([data[i]['pr_CT_RT_std_test'] for i in data.keys()][::-1])
train_pcc = np.array([data[i]['pr_CT_RT_mean_train'] for i in data.keys()][::-1])
train_pcc_std = np.array([data[i]['pr_CT_RT_std_train'] for i in data.keys()][::-1])
plt.figure(figsize=(10, 8))
plt.title('Learning Curve for CT_RT [Catboost Algorithm]',
        fontsize=18)
plt.grid(color='blue', linestyle='dotted', linewidth=0.8)
plt.xlabel("# of training data", fontsize=16)
plt.ylabel("Pearson Correlation Coefficient", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(train_dp, test_pcc, 
        c='darkblue', linestyle='-', 
        label='test score', linewidth=4)
plt.plot(train_dp, train_pcc, 
        c='darkgreen', linestyle='-', 
        label='train score', linewidth=4)
plt.fill_between(train_dp, y1=train_pcc + train_pcc_std, 
        y2=train_pcc-train_pcc_std, color='#6DAE81', alpha=0.7)
plt.fill_between(train_dp, y1=test_pcc + test_pcc_std, 
        y2=test_pcc-test_pcc_std, color='#7366BD', alpha=0.7)
plt.legend(loc='lower right', fontsize=16)
ax = plt.gca()
ax.set_xticks(train_dp)
ax2 = ax.twiny()
ax2.set_xticks(train_dp)
ax2.set_xbound(ax.get_xbound())
ax2.set_xticklabels(test_dp, fontsize=14)
ax2.set_xlabel("# of testing data", fontsize=16)
plt.tight_layout()
plt.savefig('Learning_Curve_CT_RT.png')
plt.show()
