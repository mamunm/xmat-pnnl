#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: parity_plot.py
#AUTHOR: Osman Mamun
#DATE CREATED: 02-18-2020

import numpy as np
import matplotlib.pyplot as plt

data = np.load('catboost_dict.npy', allow_pickle=True)[()]
plt.figure(figsize=(10, 8))
plt.title('Parity plot of the training and testing data for Catboost Algorithm',
        fontsize=16)
plt.grid(color='blue', linestyle='dotted', linewidth=0.8)
plt.xlabel("Experimental Rupture Time [hrs]", fontsize=16)
plt.ylabel("Predicted Ruptrue Time [hrs]", fontsize=16)
plt.plot(np.arange(0, 200000), np.arange(0, 200000), 
        c='darkblue', linewidth=4, zorder=0, label='parity line')
plt.scatter(data['CT_RT_cv_tr'], data['CT_RT_cv_tr_pred'], 
    marker='h', s=50, color='darkred', alpha=0.8, label='Train')
plt.scatter([i for i, j in zip(data['CT_RT_cv_ts'], data['CT_RT_cv_ts_pred']) 
    if j < 200000], [i for i in data['CT_RT_cv_ts_pred'] if i < 200000], 
    marker='h', s=50, color='darkgreen', alpha=0.8, label='Test')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best', fontsize=14)
ax = plt.gca()
text = r'$PCC_{0}$: {1:0.5f} $\pm$ {2:0.2f} {3} $PCC_{4}$: {5:0.5f} '
text += r'$\pm$ {6:0.2f}'
text = text.format('{test}', data['pr_CT_RT_mean_test'], 
        data['pr_CT_RT_std_test'], '\n', '{train}', 
        data['pr_CT_RT_mean_train'], data['pr_CT_RT_std_train'])
plt.text(0.2, 0.7, text,
         ha='center', va='center', fontsize=14,
         bbox=dict(facecolor='lightblue',
         edgecolor='green',
         boxstyle='round',
         pad=1),
         transform=ax.transAxes)
plt.savefig('Parity_Plot_CT_RT.png')
plt.show()
