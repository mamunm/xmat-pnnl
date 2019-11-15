#!/usr/bin/env python
# coding: utf-8

#Import different python modules
import numpy as np
import pandas as pd
import xmat_pnnl_code as xcode

#Load the data
data_df, features_description, alloy_metadata = xcode.load_data('9Cr_Data')

#Save the cleaned data into a csv for future reference
data_df.to_csv('Cleaned_data.csv', index=False)
np.save('features.npy', features_description)
alloy_metadata.to_csv('alloy_metadata.csv')





