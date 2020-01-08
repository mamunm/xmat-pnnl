#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: script_model_C.py
#AUTHOR: Osman Mamun
#DATE CREATED: 01-03-2020

import numpy as np
import pandas as pd
from xmat_pnnl_code import ConstantMatcher

# Load the csv file
df = pd.read_csv(
        '/Users/mamu867/PNNL_Code_Base/PAPML/papml_data/9_12_Cr.csv')
df = df.dropna(how='all')
df = df.dropna(subset=['CS', 'RT', 'CT Temp'])
constant_matcher = ConstantMatcher(df=df, 
        stress='CS',
        temp='CT Temp',
        RT='RT',
        groupby='ID',
        degree=1,
        logify_y=True)

np.save('C_dict.npy', constant_matcher.get_C())











