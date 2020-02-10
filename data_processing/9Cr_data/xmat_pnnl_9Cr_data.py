#!/usr/bin/env python
# coding: utf-8

#Import different python modules
import numpy as np
import pandas as pd
import xmat_pnnl_code as xcode
import mendeleev

#Load the data
data_df, features_description, alloy_metadata = xcode.load_data(
        data_file='9Cr_Data')

#Get the weighted atomic number for each alloy
ele = [k for k, v in features_description.items() if 'Element' in v]
AN = {k: getattr(mendeleev, k).atomic_number for k in ele}
AW = {k: getattr(mendeleev, k).atomic_weight for k in ele}
BP = {k: getattr(mendeleev, k).boiling_point for k in ele}
DP = {k: getattr(mendeleev, k).dipole_polarizability for k in ele}
EH = {k: getattr(mendeleev, k).evaporation_heat for k in ele}
HF = {k: getattr(mendeleev, k).heat_of_formation for k in ele}
MP = {k: getattr(mendeleev, k).melting_point for k in ele}

data_df['Weighted_AN'] = data_df[ele].mul(AN, axis=1).sum(axis=1)/100
data_df['Weighted_AW'] = data_df[ele].mul(AW, axis=1).sum(axis=1)/100
data_df['Weighted_BP'] = data_df[ele].mul(BP, axis=1).sum(axis=1)/100
data_df['Weighted_DP'] = data_df[ele].mul(DP, axis=1).sum(axis=1)/100
data_df['Weighted_EH'] = data_df[ele].mul(EH, axis=1).sum(axis=1)/100
data_df['Weighted_HF'] = data_df[ele].mul(HF, axis=1).sum(axis=1)/100
data_df['Weighted_MP'] = data_df[ele].mul(MP, axis=1).sum(axis=1)/100
#Save the cleaned data into a csv for future reference
data_df.to_csv('Cleaned_data.csv', index=False)
np.save('features.npy', features_description)
alloy_metadata.to_csv('alloy_metadata.csv')





