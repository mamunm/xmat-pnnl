#!/usr/bin/env python
# -*-coding: utf-8 -*-

#!/bin/bash
#SBATCH -N 1
#SBATCH -C knl
#SBATCH -q debug
#SBATCH -J grid_search
#SBATCH --mail-user=mdosman.mamun@pnnl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#SCRIPT: script.py
#AUTHOR: Osman Mamun
#DATE CREATED: 12-19-2019

import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from xmat_pnnl_code import ProcessData
import xmat_pnnl_code as xcode
from sklearn.model_selection import train_test_split

#Model data
base_path = '/'.join(xcode.__path__[0].split('/')[:-1])
path = base_path + '/data_processing/9Cr_data/LMP'
model = np.load(path + '/model_params.npy', allow_pickle=True)[()]
model = model['9Cr-001']

#Load the 9Cr data
ID = [1, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
      43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60,
      61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
      77, 78, 79, 82]
ID = ['9Cr-{}'.format(str(i).zfill(3)) for i in ID]

path = base_path + '/data_processing/9Cr_data'
df = pd.read_csv(path + '/Cleaned_data.csv')
df = df[df.ID.isin(ID)]
ele = ['Fe', 'C', 'Cr', 'Mn', 'Si', 'Ni', 'Co', 'Mo', 'W', 'Nb', 'Al',
       'P', 'Cu', 'Ti', 'Ta', 'Hf', 'Re', 'V', 'B', 'N', 'O', 'S', 'Zr']
df[ele] = df[ele].fillna(0)
df = df.dropna(subset=['CT_RT', 'CT_CS', 'CT_EL', 'CT_RA', 'CT_Temp',
    'Normal', 'Temper1', 'AGS No.', 'CT_MCR'])
df['log_CT_CS'] = np.log(df['CT_CS'])
df['log_CT_MCR'] = np.log(df['CT_MCR'])

features = [i for i in df.columns if i not in ['CT_CS', 
    'CT_MCR', 'ID']]
X = df[features].to_numpy(np.float32)
#print(X.shape)
y = df['CT_RT'].to_numpy(np.float32)

pdata = ProcessData(X=X, y=y, features=features)
#pdata.clean_data(scale_strategy={'strategy': 'power_transform', 
#    'method': 'yeo-johnson'})
pdata.clean_data(scale_strategy={'strategy': 'MinMaxScaler'})
data = pdata.get_data()
scale = pdata.scale
#X = scale.inverse_transform(data['X'])
X = data['X']
features = data['features']
del data, pdata, df, y, ID, ele, model

X_train, X_test = train_test_split(X, test_size=0.1)


#Variational Autoencoder
from scipy.stats import norm
from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential

original_dim = 33
intermediate_dim = 22
latent_dim = 11
batch_size = 100
epochs = 100
epsilon_std = 1.0


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
'''
from keras.losses import mse

def nll(y_true, y_pred):
    return mse(y_true, y_pred)
'''
class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid')
])

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

vae = Model(inputs=[x, eps], outputs=x_pred)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9,
        beta_2=0.999, amsgrad=False)
#vae.compile(optimizer='rmsprop', loss=nll)
vae.compile(optimizer=adam, loss=nll)

hist = vae.fit(X_train,
        X_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, X_test))

pd.DataFrame(hist.history).plot()
plt.show()
encoder = Model(x, z_mu)

'''
# display a 2D plot of the digit classes in the latent space
z_test = encoder.predict(X_test, batch_size=batch_size)

# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))
z_grid = norm.ppf(u_grid)
x_decoded = decoder.predict(z_grid)
'''
X_pred = vae.predict(X_test) 
print(X_test[-1], X_pred[-1])
X_test = scale.inverse_transform(X_test)
X_pred = scale.inverse_transform(X_pred)
for i, j in zip(X_test[-1], X_pred[-1]):
    print(i, j, 100*(i-j)/i)

z1 = norm.ppf(np.linspace(0.01, 0.99, 10))
z2 = norm.ppf(np.linspace(0.01, 0.99, 10))
z_grid = np.dstack(np.meshgrid([norm.ppf(np.linspace(0.01, 0.99, 10)) 
    for _ in range(110)]))

x_pred_grid = decoder.predict(z_grid.reshape(10*10, latent_dim))
print(scale.inverse_transform(x_pred_grid))
