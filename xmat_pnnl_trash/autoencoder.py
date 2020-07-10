from keras.layers import Lambda, Input, Dense, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras import optimizers

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

class AutoEncoder:
    
    def __init__(self, arch=None, X=None, loss='mse', 
            epochs=1000, batch_size=100, optimizer='rmsprop', 
            opt_dict=None, epsilon_std = 1.0):
        self.arch = arch
        self.X = X
        self.loss = loss
        self.epochs = epochs
        self.batch_size =  batch_size
        self.optimizer = optimizer
        self.opt_dict = opt_dict
        self.epsilon_std = epsilon_std

    def mse(self, y_true, y_pred):
        return mse(y_true, y_pred)

    def bce(self, y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    def build_model(self):
        original_dim = self.arch[0]
        intermediate_dim = self.arch[1:-1]
        latent_dim = self.arch[-1]
        decoder_layers = [Dense(intermediate_dim[-1], input_dim=latent_dim, 
                activation='relu')]
        odm = intermediate_dim[-1]
        for idm in intermediate_dim[::-1][1:]:
            decoder_layers += [Dense(idm, input_dim=odm, activation='relu')]
            odm = idm
        decoder_layers +=[Dense(original_dim, activation='sigmoid')]
        self.decoder = Sequential(decoder_layers)
        x = Input(shape=(original_dim,))
        for n, idm in enumerate(intermediate_dim):
            if n == 0:
                h = Dense(idm, activation='relu')(x)
            else:
                h = Dense(idm, activation='relu')(h)

        z_mu = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
        eps = Input(tensor=K.random_normal(stddev=self.epsilon_std,
            shape=(K.shape(x)[0], latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])
        x_pred = self.decoder(z)
        self.vae = Model(inputs=[x, eps], outputs=x_pred)
        
        if self.optimizer == 'adam' and self.opt_dict == None:
            opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                    beta_2=0.999, amsgrad=False)
        elif self.optimizer == 'adam':
            opt = optimizers.Adam(**self.opt_dict)

        elif self.optimizer == 'sgd' and self.opt_dict == None:
            opt = optimizers.SGD(lr=0.01, decay=1e-6, 
                    momentum=0.9, nesterov=True)
        
        elif self.optimizer == 'sgd':
            opt = optimizers.SGD(**self.opt_dict)
        
        else:
            opt = 'rmsprop'

        if self.loss == 'mse':
            self.vae.compile(optimizer=opt, loss=mse)
        if self.loss == 'bce':
            self.vae.compile(optimizer=opt, loss=bce)

        X_train, X_test = train_test_split(self.X, test_size=0.1)
        self.hist = self.vae.fit(X_train,
                                 X_train,
                                 shuffle=True,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(X_test, X_test))
    
    def predict(self, X):
        return self.vae.predict(X)

    '''
    def get_random_alloy(self, n_sample_per_direction=10):
        latent_dim = self.arch[-1]
        z_grid = np.dstack(np.meshgrid([norm.ppf(
            np.linspace(0.01, 0.99, n_sample_per_direction)) 
            for _ in range(n_sample_per_direction * latent_dim)]))
        return self.decoder.predict(z_grid.reshape(
            n_sample_per_direction*n_sample_per_direction, latent_dim))
    '''

    def get_random_alloy(self, n_sample_per_direction=10):
        return self.decoder.predict(np.random.normal(0, 1, 
            size=(n_sample_per_direction, self.arch[-1])))
    
    def get_linspace_alloy(self, n_sample_per_direction=10):

        return self.decoder.predict(np.array(np.meshgrid(
            [np.linspace(0, 1, n_sample_per_direction) 
                for _ in range(self.arch[-1])])).T.reshape(-1, self.arch[-1]))

    def get_hist_plot(self):
        return pd.DataFrame(self.hist.history).plot()

    def get_hist(self):
        return pd.DataFrame(self.hist.history)

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
