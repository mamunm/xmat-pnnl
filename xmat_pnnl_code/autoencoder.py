from keras.layers import Lambda, Input, Dense, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras import optimizers

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

class AutoEncoder:
    
    def __init__(self, arch=None, X=None, loss='mse', 
            epoch=1000, batch_size=100, optimizer='rmsprop', opt_dict=None):
        self.arch = arch
        self.X = X
        self.loss = loss
        self.epochs = epochs
        self.batch_size =  batch_size
        self.optimizer = optimizer
        self.opt_dict = opt_dict

    def mse(self, y_true, y_pred):
        return mse(y_true, y_pred)

    def bce(self, y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    def build_model(self):
        original_dim = self.arch[0]
        intermediate_dim = self.arch[1]
        latent_dim = self.arch[2]
        self.decoder = Sequential([
            Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
            Dense(original_dim, activation='sigmoid')])
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
        self.vae = Model(inputs=[x, eps], outputs=x_pred)
        
        if self.optimizer = 'adam' and self.opt_dict == None:
            opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                    beta_2=0.999, amsgrad=False)
        elif self.optimizer = 'adam':
            opt = optimizers.Adam(**self.opt_dict)

        elif self.optimizer = 'sgd' and self.opt_dict == None:
            opt = optimizers.SGD(lr=0.01, decay=1e-6, 
                    momentum=0.9, nesterov=True)
        
        elif self.optimizer = 'sgd':
            opt = optimizers.SGD(**self.opt_dict)
        
        else:
            opt = 'rmsprop'

        self.vae.compile(optimizer=opt, loss=nll)
        X_train, X_test = train_test_split(X, test_size=0.1)
        self.hist = self.vae.fit(X_train,
                                 X_train,
                                 shuffle=True,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(X_test, X_test))
    
    def predict(self, X):
        return self.vae.predict(X)

    def get_random_alloy(self, n_sample_per_direction=10):
        latent_dim = self.arch[-1]
        z_grid = np.dstack(np.meshgrid([norm.ppf(
            np.linspace(0.01, 0.99, n_sample_per_direction)) 
            for _ in range(n_sample_per_direction * latent_dim)]))
        returnm decoder.predict(z_grid.reshape(
            n_sample_per_direction*n_sample_per_direction, latent_dim))

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
