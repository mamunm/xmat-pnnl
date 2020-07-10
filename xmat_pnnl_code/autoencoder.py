import keras
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class AutoEncoder:
    def __init__(self, X=None, 
                       train_size=0.8, 
                       arch=None,
                       loss='xent',
                       epochs=1000,
                       batch_size=128,
                       optimizer='adam',
                       weights=None,
                       ):
        self.X_train, self.X_test = train_test_split(X, train_size=train_size)
        self.arch = arch
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.weights = weights

    def build_model(self):
        # build encoder model
        inputs = Input(shape=(self.X_train.shape[1], ), name='encoder_input')
        x = Dense(self.arch[1], activation='relu')(inputs)
        if len(self.arch) == 4:
            x = Dense(self.arch[2], activation='relu')(x)
        z_mean = Dense(self.arch[-1], name='z_mean')(x)
        z_log_var = Dense(self.arch[-1], name='z_log_var')(x)
        z = Lambda(sampling, output_shape=(self.arch[-1],), 
                name='z')([z_mean, z_log_var])
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()
        #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
        # build decoder model
        latent_inputs = Input(shape=(self.arch[-1],), name='z_sampling')
        if len(self.arch) == 4:
            x = Dense(self.arch[2], activation='relu')(latent_inputs)
            x = Dense(self.arch[1], activation='relu')(x)
        else:
            x = Dense(self.arch[1], activation='relu')(latent_inputs)

        outputs = Dense(self.X_train.shape[1], activation='sigmoid')(x)
        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        if self.loss not in ["mse", "xent"]:
            raise KeyError("The loss function is not available.")
        if self.loss == "mse":
            reconstruction_loss = mse(inputs, outputs)
        if self.loss == "xent":
            reconstruction_loss = binary_crossentropy(inputs, outputs)

        reconstruction_loss *= self.X_train.shape[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=self.optimizer)
        
        # train the autoencoder
        if not self.weights:
            self.vae.fit(self.X_train,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        validation_data=(self.X_test, None))
        else:
            self.vae.load_weights(self.weights + ".h5")

    def save_model(self, f_name):
        self.vae.save_weights(f_name + ".h5")

    def predict(self, X):
        return self.vae.predict(X)

    def get_random_alloy(self, n_samples=400):
        return self.decoder.predict(np.random.normal(0, 1,
            size=(n_samples, self.arch[-1])))

    def get_linspace_alloy(self, n_range=(-1, 1), n_sample_per_direction=10):

        return self.decoder.predict(np.array(np.meshgrid(
            [np.linspace(*n_range, n_sample_per_direction)
                for _ in range(self.arch[-1])])).T.reshape(-1, self.arch[-1]))

    def get_hist_plot(self):
        return pd.DataFrame(self.hist.history).plot()

    def get_hist(self):
        return pd.DataFrame(self.hist.history)
