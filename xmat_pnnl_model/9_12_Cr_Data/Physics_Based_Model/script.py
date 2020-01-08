#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: script.py
#AUTHOR: Osman Mamun
#DATE CREATED: 01-03-2020

import numpy as np
import pandas as pd
from papml_code import AlloyDataPreper

fillvals = {'Fe': 0,'C':0,'Cr':0,'Mn':0,
            'Si':0,'Ni':0,'Co':0,'Mo':0,
            'W':0,'Nb':0,'Al':0,'P':0,
            'Cu':0,'Ti':0,'Ta':0,'Hf':0,
            'Re':0,'V':0,'B':0,'N':0,
            'O':0,'S':0,'Homo':0,'Normal':25,
            'Temper1':25,'Temper2':25,'Temper3':25}
dropna9_12Cr = ['ID','CT Temp','CS','RT','AGS','AGS No.']
exclude9_12Cr = ['TT Temp','YS',
                  'UTS','Elong','RA','EL','RA_2',
                  'MCR','0.1% CS','0.2% CS','0.5% CS',
                  '1.0% CS','2.0% CS','5.0% CS','TTC',
                  'Temper3','ID','Hf','Homo','Re','Ta','Ti','O']
N9_12Cr = AlloyDataPreper(Dataset='9_12_Cr.csv',
                         label='RT',
                         dropna_cols=dropna9_12Cr,
                         exclude_cols=exclude9_12Cr,
                         fill_vals=fillvals)
ready9_12Cr = N9_12Cr.prep_it()
X = ready9_12Cr['preds']
y = ready9_12Cr['labels']
metadata = ready9_12Cr['preds'].columns
del ready9_12Cr
print(y)



'''
pd = ProcessData(X=X, y=y, metadata=metadata)
pd.clean_data()
X, y, metadata = pd.get_data()
del pd
'''
