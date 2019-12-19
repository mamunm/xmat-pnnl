#!/usr/bin/env python
# -*-coding: utf-8 -*-

#SCRIPT: script.py
#AUTHOR: Osman Mamun
#DATE CREATED: 12-16-2019

import pandas as pd
import numpy as np
import mendeleev

path = '/Users/mamu867/PNNL_Code_Base/xmat-pnnl/data_processing/9Cr_data'
df = pd.read_csv(path + '/Cleaned_data.csv')
features_description = np.load(path + '/features.npy', allow_pickle=True)[()]

synthetic_features = ['atomic_radius',
        'atomic_volume',
        'atomic_weight',
        'boiling_point',
        'covalent_radius_cordero',
        'covalent_radius_pyykko',
        'covalent_radius_slater',
        'density',
        'dipole_polarizability',
        'electron_affinity',
        'en_allen',
        'en_ghosh',
        'en_pauling',
        'evaporation_heat',
        'fusion_heat',
        'heat_of_formation',
        'specific_heat',
        'vdw_radius',
        'melting_point',
        'metallic_radius']

ele = [k for k, v in features_description.items() if 'Element' in v]

for sf in synthetic_features:
    f = {k: getattr(getattr(mendeleev, k), sf) for k in ele}
    df['Weighted_{}'.format(sf)] = df[ele].mul(f, axis=1).sum(axis=1)/100

df.to_csv('fingerprints_data.csv', index=False)
