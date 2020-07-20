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
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load('data.npy', allow_pickle=True)[()]
df = pd.DataFrame(data['y_real'], columns=['rupture life'])
df['sample'] = 'real'
df = df.append(pd.DataFrame({'rupture life': data['y_gen'], 'sample':
    ['synthetic' for i in range(len(data['y_gen']))]}), ignore_index=True)
ax1 = sns.violinplot(y='rupture life', x='sample', data=df)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Rupture Life', fontsize=16)
plt.xlabel('')
plt.tight_layout()
plt.savefig('violin_plot_Aus.png')
plt.show()


