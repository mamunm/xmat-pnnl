#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: plot.py
#AUTHOR: Osman Mamun
#DATE CREATED: 07-07-2020

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = np.load('val_data_chrome_1.npy', allow_pickle=True)[()]

pca_gen = PCA(n_components=2).fit_transform(data['X_gen'])
pca_real = PCA(n_components=2).fit_transform(data['X_real'])

#PCA plot before applying clustering
plt.figure()
plt.title('Principal component analysis plot', fontsize=16)
plt.grid(color='blue', linestyle='dotted', linewidth=0.8)
plt.xlabel(r"$X_1$", fontsize=16)
plt.ylabel(r"$X_2$", fontsize=16)
plt.scatter(pca_gen[:, 0], pca_gen[:, 1], marker='h', s=50,
        color='darkred', alpha=0.8, label='Generated samples')
plt.scatter(pca_real[:, 0], pca_real[:, 1], marker='h', s=50,
        color='darkblue', alpha=0.8, label='Real samples')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best', fontsize=14)
plt.show()
