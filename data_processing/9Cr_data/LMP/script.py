#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: script.py
#AUTHOR: Osman Mamun
#DATE CREATED: 12-10-2019

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.simplefilter("ignore", UserWarning)
import matplotlib.pyplot as plt

from xmat_pnnl_code import PolyFit
from xmat_pnnl_code import Shapley

ID = [1, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
      43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60,
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
      77, 78, 79, 80, 82]
ID = ['9Cr-{}'.format(str(i).zfill(3)) for i in ID]

data = pd.read_csv('../Cleaned_data.csv')
keep_column = ['ID', 'CT_Temp', 'CT_CS', 'CT_RT']
data = data[keep_column]
C = 25
data['LMP'] = 1e-3 * (data['CT_Temp']) * (np.log(data['CT_RT']) + C)
score_lib = {}
model_lib = {}

for alloy_id in ID:
    df = data[data['ID'] == alloy_id]
    poly = PolyFit(df=df[['LMP', 'CT_CS']], 
            target='CT_CS', 
            degree=1,
            logify_y=True)
    poly.fit()
    print('Score for {}: {}'.format(alloy_id, poly.score))
    score_lib[alloy_id] = poly.score
    poly.plot_model().savefig('poly_fit_curves/{}.png'.format(alloy_id))
    plt.clf()
    model_lib[alloy_id] = {'coef': poly.model.coef_, 
            'intercept': poly.model.intercept_}
    
    np.save('poly_score.npy', score_lib)
    np.save('model_params.npy', model_lib)

    '''
    # Run shapley on the data
    shapley = Shapley(df=df[['LMP', 'CT_CS']], target='CT_CS', degree=2)
    phi, phi_percentage = shapley.get_phi()
    df['shapley'] = phi
    df['shapley_percentage'] = phi_percentage
    print(df)
    df.to_csv('shapley_values/{}.csv'.format(alloy_id))
    '''
