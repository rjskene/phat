"""
Gaussian-based neural network
"""
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from phat.learn.utils import GraphicMixin, nnelu

def mon_mean(y, y_):
    mu, sigma = tf.unstack(y_, num=2, axis=-1)
    return tf.keras.backend.mean(mu)

def mon_std(y, y_):
    mu, sigma = tf.unstack(y_, num=2, axis=-1)
    return tf.keras.backend.mean(sigma)

def gnll_loss(y, y_):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    mu, sigma = tf.unstack(y_, num=2, axis=-1)
    dist = tfd.Normal(mu, sigma)
    return tf.reduce_mean(-dist.log_prob(y))

class DN(tf.keras.Model, GraphicMixin):
    def __init__(self, neurons=200):
        tf.keras.backend.set_floatx('float64')
        tf.keras.utils.get_custom_objects().update({
            'nnelu': tf.keras.layers.Activation(nnelu),
            'mean': mon_mean,
            'std': mon_std
        })
        super(DN, self).__init__(name="DN")
        self.neurons = neurons
        
        self.h1 = tf.keras.layers.Dense(neurons, input_shape=(1,), activation='tanh', name='h1')
        
        self.mu = tf.keras.layers.Dense(1, name="mu")
        self.sigma = tf.keras.layers.Dense(1, activation="nnelu", name="sigma")
        self.pvec = tf.keras.layers.Concatenate(name="pvec")

    def call(self, inputs):
        x = self.h1(inputs)
        
        mu_v = self.mu(x)
        sigma_v = self.sigma(x)
        
        return self.pvec([mu_v, sigma_v])
