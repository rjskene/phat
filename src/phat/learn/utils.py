from typing import Union, Iterable
import numpy as np
import functools
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

from sklearn.model_selection import train_test_split
import tensorflow as tf

from phat.utils import arrayarize

def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float64), tf.nn.elu(input))

def asymmactiv(x):
    lam = tf.constant(1, dtype=tf.float64)
    term1 = tf.constant(1, dtype=tf.float64) + tf.math.multiply(lam, tf.math.exp(x))
    return tf.constant(1, dtype=tf.float64) - term1**(-1./lam)

def splitter(func):
    @functools.wraps(func)
    def wrap(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        return func(self, x, y, sample_weight)
    
    return wrap

class GraphicMixin:
    def build_graph(self):
        inputs = tf.keras.Input(shape=(1))
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))

    def plot_model(self, show_shapes:bool=True, to_file:str='phatdn.png', *args, **kwargs):
        return tf.keras.utils.plot_model(
            self.build_graph(),
            to_file=to_file,
            show_shapes=show_shapes
        )

    def loss_progress(self, history=None, metric='', ax=None):
        history = self.history if history is None else history
        
        if ax is None:
            fig, ax = plt.subplots()

        C2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
        C2_rgba = np.ones(4)
        C2_rgba[:3] = mcolors.to_rgb(C2)
        C2_rgba[3] = 0
        dots = plt.scatter([0], [0], marker='o', color=C2_rgba, s=100)
        vloss = np.array(history.history['val_loss'])
        vloss = vloss[~np.isnan(vloss)]

        metric = metric if metric else 'val_mean'
        vmean = np.array(history.history[metric])
        vmean = vmean[~np.isnan(vloss)]

        mumin, mumax = np.array(vmean).min(), \
            np.array(vmean).max()
        # stdmin, stdmax = np.array(history.history['val_std']).min(), \
        #     np.array(history.history['val_std']).max()
        lossmin, lossmax = np.array(vloss).min(), \
            np.array(vloss).max()

        # ms = np.linspace(mumin,mumax)
        # stds = np.linspace(stdmin,stdmax)
        ax.axvline(vmean[-1], c='C4', ls='--', lw=2)

        ax.set_xlim((
            mumin*1.1 if mumin <=0 else mumin*.9, 
            mumax*1.1 if mumax >=0 else mumax*.9,
        ))    
        ax.set_ylim((
            lossmin*1.1 if lossmin <=0 else lossmin*.9, 
            lossmax*1.1 if lossmax >=0 else lossmax*.9,
        ))

        datetext = ax.text(
            0, 1.1, '', 
            size=16, alpha=1, fontweight='bold', 
            ha='left', va='top', transform=ax.transAxes
        )

        def snap(i):
            alphas = np.tanh(np.linspace(3,0,i+1))
            if alphas.size > 1:     
                alphas[1:] /= 8

            colors = np.zeros((i+1, 4))
            colors[:i+1, :3] = C2_rgba[:3]
            colors[:i+1, 3] = alphas[::-1]

            vals = np.vstack((
                vmean[:i+1], 
                vloss[:i+1]
            )).T
            dots.set_offsets(vals)
            dots.set_color(colors)
            datetext.set_text(f'Epoch {i}')

            return dots

        anime = FuncAnimation(fig, snap, blit=False, 
            frames=history.epoch, 
            interval=200, repeat=False
        )
        return anime

class DataSplit:
    #     https://www.baeldung.com/cs/data-normalization-before-after-splitting-set
    # Split validaiton set, THEN apply normalizaiton and then split test set
    XY = namedtuple('XY', ['x', 'y'])

    def __init__(self, 
        y, x:Iterable[float]=None, 
        test_size:float=.1, val_size:float=.1,
        batch_sizes:Union[int, Iterable]=32,
        preprocess:str=''
        ):
        train_batch, test_batch, val_batch = self._set_batch_sizes(batch_sizes)

        y = arrayarize(y)
        x = arrayarize(x) if x is not None else np.zeros(y.size)
        self.raw = self.XY(x.copy(), y.copy())
        self.test_size = test_size
        self.val_size = val_size

        x, xt, y, yt = train_test_split(x.reshape(-1,1), y.reshape(-1,1), test_size=val_size, random_state=42)
        
        if preprocess == 'minmax':
            y = self.rescale(y)
        elif preprocess == 'minmaxminus':
            y = 2*self.rescale(y) - 1
        elif preprocess == 'allpos':
            y = self.pos_shift(y)
        elif preprocess == 'tanh':
            y = np.tanh(y)

        x, xval, y, yval = train_test_split(x, y, test_size=test_size, random_state=42)
        
        self.train_raw = self.XY(x.copy(), y.copy())
        self.train = tf.data.Dataset.from_tensor_slices((x, y)).batch(train_batch)
        self.test_raw = self.XY(xt.copy(), yt.copy())
        self.test = tf.data.Dataset.from_tensor_slices((xt, yt)).batch(test_batch)
        self.val_raw = self.XY(xval.copy(), yval.copy())
        self.val = tf.data.Dataset.from_tensor_slices((xval, yval)).batch(val_batch)

    def _set_batch_sizes(self, batch_sizes):
        if isinstance(batch_sizes, (float, int)):
            return list([batch_sizes])*3
        elif isinstance(batch_sizes, (list, tuple)):
            if len(batch_sizes) == 3:
                return batch_sizes
            elif len(batch_sizes) == 2:
                batch_sizes = list(batch_sizes)
                batch_sizes.append(32)
                return batch_sizes
            else:
                raise ValueError('`batch_sizes` can only be length 2/3')
        else:
            raise ValueError('`batch_sizes` can be int, list, or dict')

    def rescale(self, y):
        num = y - y.min()
        denom = y.max() - y.min()
        return num / denom

    def descale(self, y_scaled):
        return (y_scaled*(self.raw.y.max() - self.raw.y.min())) + self.raw.y.min()

    def pos_shift(self, y):
        y_ = y - y.min() if y.min() < 0 else y
        return y_
