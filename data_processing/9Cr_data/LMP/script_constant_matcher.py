#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: script_constant_matcher.py
#AUTHOR: Osman Mamun
#DATE CREATED: 12-13-2019

import numpy as np
import pandas as pd
from xmat_pnnl_code import ConstantMatcher

path = '/Users/mamu867/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data'
data = pd.read_csv(path + '/Cleaned_data.csv')
ID = [1, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
      43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60,
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
      77, 78, 79, 80, 82]
ID = ['9Cr-{}'.format(str(i).zfill(3)) for i in ID]
cm = ConstantMatcher(df=data, 
        groupby=None, 
        ID_list=ID, 
        target='CT_CS', 
        degree=1,
        logify_y=True)
score_lib = cm.get_C()

np.save('constant_matcher_score_lib.npy', score_lib)
