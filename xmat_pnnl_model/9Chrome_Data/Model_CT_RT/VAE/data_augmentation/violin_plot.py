#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: violin_plot.py
#AUTHOR: Osman Mamun
#DATE CREATED: 07-09-2020

import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import pandas as pd

data = np.load('catboost_res.npy', allow_pickle=True)[()]
y_real = data['scale'].inverse_transform(data['X'])[:, -1]
y_gen = np.concatenate([i[:, -1] for i in data['gen_sample']])
df = pd.DataFrame(y_real, columns=['rupture life'])
df['sample'] = 'real'
df = df.append(pd.DataFrame({'rupture life': y_gen, 'sample': 
    ['synthetic' for i in range(len(y_gen))]}), ignore_index=True)
ax1 = sns.violinplot(y='rupture life', x='sample', data=df)
plt.show()

