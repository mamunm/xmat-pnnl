#!/usr/bin/env python
# coding: utf-8

#Import modules
import numpy as np
import pandas as pd
import xmat_pnnl_code as xcode
import mendeleev 

#Load the data
data_df, features_description, alloy_metadata = xcode.load_data('Aus_Steel_Data')

#Get the weighted atomic number for each alloy
ele = [k for k, v in features_description.items() if 'Element' in v]
AN = {k: getattr(mendeleev, k).atomic_number for k in ele}
data_df['Weighted_AN'] = data_df[ele].mul(AN, axis=1).sum(axis=1)/100

#Save the cleaned data into a csv for future reference
data_df.to_csv('Cleaned_data.csv', index=False)
np.save('features.npy', features_description)
alloy_metadata.to_csv('alloy_metadata.csv')

