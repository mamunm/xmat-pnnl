#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: script_shapley.py
#AUTHOR: Osman Mamun
#DATE CREATED: 12-18-2019

import numpy as np
from itertools import combinations
from xmat_pnnl_code import ShapleyFeatures
import pandas as pd

data = np.load('gp_run.npy', allow_pickle=True)[()]

df = pd.DataFrame(data['X'], columns=data['features'])
df['C'] = data['y']

shapley_information = ShapleyFeatures(df=df, 
        target='C', features=data['features']).get_phi()

np.save('shpaley_information.npy', shapley_information)
    



