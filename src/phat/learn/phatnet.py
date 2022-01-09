import numpy as np
import pandas as pd
import itertools as it

import tensorflow as tf

from phat.learn.dists import Phat4TF
from phat.learn.utils import GraphicMixin, splitter

logger = tf.get_logger()

class PhatMetric(tf.keras.metrics.Metric):
    """
    Custom tensorflow metric that updates the Phat distribution parameters
    """
 
    param_names = ['mean', 'std', 'loc', 'scale', 'shape']
    ends = ['right', 'left']
    ALLOWED_NAMES = ['_'.join(tup) for tup in it.product(param_names, ends)]
    
    def __init__(self, name, **kwargs):
        if not name in self.ALLOWED_NAMES:
            raise ValueError(f"`name` must be one of {', '.join(self.ALLOWED_NAMES)}")
        super(PhatMetric, self).__init__(name=name, **kwargs)
        self.param = self.add_weight(name='param', initializer='zeros')
        
    def update_state(self, y, phat, sample_weight=None):
        p_name = self.name.split('_')
        param = getattr(getattr(phat, p_name[1]), p_name[0])
        self.param.assign(tf.reduce_mean(param))

    def result(self):
        return self.param

    def reset_state(self):
        self.param.assign(0)

class AMLSE(tf.keras.metrics.Metric):
    ALLOWED_NAMES = ['left', 'right']
    def __init__(self, shape, name, **kwargs):
        if not name in self.ALLOWED_NAMES:
            raise ValueError(f"`name` must be one of {', '.join(self.ALLOWED_NAMES)}")        
        super(AMLSE, self).__init__(name=f'amlse_{name}', **kwargs)
        self.shape = shape
        self.tail = self.name[5:]
        self.amlse = self.add_weight(name='amlse', initializer='zeros')

    @staticmethod
    def calc_amlse(shape_, shape):
        return tf.reduce_mean((tf.math.log(shape_) - tf.math.log(shape))**2)
        
    @staticmethod
    def shapes_for_tail(phat, tail, shape):
        shape_ = getattr(phat, tail).shape
        shape = tf.ones_like(shape_)*shape
        return shape_, shape
    
    @staticmethod
    def calc(phat, tail, shape):
        shape_, shape = AMLSE.shapes_for_tail(phat, tail, shape)
        return AMLSE.calc_amlse(shape_, shape)
    
    def update_state(self, y, phat, sample_weight=None):
        self.amlse.assign(AMLSE.calc(phat, self.tail, self.shape))
                          
    def result(self):
        return self.amlse

    def reset_state(self):
        self.amlse.assign(0)

class TwoTailedAMLSE(tf.keras.metrics.Metric):
    def __init__(self, shape_l, shape_r, name='two_tailed_amlse', **kwargs):
        super(TwoTailedAMLSE, self).__init__(name=name, **kwargs)
        self.shape_l = shape_l
        self.shape_r = shape_r       
        self.two_tailed = self.add_weight(name='two_tailed_amlse', initializer='zeros')

    @staticmethod
    def calc(phat, shape_l, shape_r):
        left = AMLSE.calc(phat, 'left', shape_l)
        right = AMLSE.calc(phat, 'right', shape_r)
        return tf.reduce_mean([left,right])

    def update_state(self, y, phat, sample_weight=None):
        self.two_tailed.assign(TwoTailedAMLSE.calc(phat, self.shape_l, self.shape_r))
        
    def result(self):
        return self.two_tailed

    def reset_state(self):
        self.two_tailed.assign(0)

class NLL(tf.keras.metrics.Metric):
    def __init__(self, name='nll', **kwargs):
        super(NLL, self).__init__(name=name, **kwargs)
        self.nll = self.add_weight(name='nll', initializer='zeros')

    @staticmethod
    def calc_nll(y_, axis=None):
        """
        """
        return tf.reduce_mean(y_, axis=axis)

    @staticmethod
    def calc(y, phat, axis=None):
        return NLL.calc_nll(-phat.log_prob(y), axis=axis)
        
    def update_state(self, y, phat, sample_weight=None):
        self.nll.assign(NLL.calc(y, phat))
                          
    def result(self):
        return self.nll

    def reset_state(self):
        self.nll.assign(0)

class PhatLossMetric(tf.keras.metrics.Metric):
    def __init__(self, *args, name='phat_loss_metric', **kwargs):
        super(PhatLossMetric, self).__init__(name=name, **kwargs)
        self.phat_metric = self.add_weight(name='phat_metric', initializer='zeros')
        
        self.nll = NLL()
        self.tails = TwoTailedAMLSE(*args, **kwargs)

    def update_state(self, y, phat, sample_weight=None):
        nll = self.nll.calc(y, phat)
        tails = self.tails.calc(phat, self.tails.shape_l, self.tails.shape_r)
        self.phat_metric.assign(nll / (tails + 1))
                          
    def result(self):
        return self.phat_metric

    def reset_state(self):
        self.phat_metric.assign(0)
        
class BodyLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(BodyLoss, self).__init__(name="BodyLoss")
        
    def call(self, y, y_):
        return NLL.calc_nll(y_)

class TailLoss(tf.keras.losses.Loss):
    def __init__(self, shape_l, shape_r):
        super(TailLoss, self).__init__(name="TailLoss")
        self.shape_l = shape_l
        self.shape_r = shape_r
        
        self.shapes = tf.constant([[shape_l, shape_r]], dtype=tf.float64)
        
    def call(self, y, shapes_):
        shapes = tf.tile(self.shapes, tf.shape(y))
        loss = AMLSE.calc_amlse(shapes_, shapes)
        return loss
    
class PhatLoss(tf.keras.losses.Loss):
    """
    Custom loss function for PHAT model assessment
    
    Convex combination of:
    1) the negative likelihood
    2) RMSE of extreme quantitles
        > ensures fit in the tails
    """
    def __init__(self, *args, **kwargs):
        super(PhatLoss, self).__init__(name="PhatLoss")
        self.BodyLoss = BodyLoss()
        self.TailLoss = TailLoss(*args, **kwargs)

    def call(self, y, out, sample_weight=None):
        y_ = out[:,0]
        shapes_ = out[:,1:]
        body_loss = self.BodyLoss(y, y_)
        tail_loss = self.TailLoss(y, shapes_)
        return body_loss / (tail_loss + 1)

class PhatNetBeta(tf.keras.Model, GraphicMixin):
    def __init__(self, neurons=4):
        tf.keras.backend.set_floatx('float64')
        super(PhatNetBeta, self).__init__(name='PhatNet')
        
        self.h1 = tf.keras.layers.Dense(neurons, input_shape=(1,), name='h1')

        self.mu = tf.keras.layers.Dense(1, name='mu')
        self.sigma = tf.keras.layers.Dense(1, activation='softplus', name='sigma')
        self.shape_l = tf.keras.layers.Dense(1, activation='softplus', name='shape_left')
        self.shape_r = tf.keras.layers.Dense(1, activation='softplus', name='shape_right')        

        self.vec = tf.keras.layers.Concatenate(name='vec')
                
    def call(self, xy):
        x = self.h1(xy)
        
        mu_v = self.mu(x)
        sigma_v = self.sigma(x)
        shape_l_v = self.shape_l(x)
        shape_r_v = self.shape_r(x)

        vec = self.vec([mu_v, sigma_v, shape_l_v, shape_r_v])
        return vec

    @splitter
    def train_step(self, x, y, sample_weight):            
        with tf.GradientTape() as tape:
            vec = self(x, training=True)
            mean_, std_, shape_l_, shape_r_ = tf.unstack(vec, num=4, axis=-1)
        
            phat_ = Phat4TF(mean_, std_, shape_l_, shape_r_)
            y_ = NLL.calc(y, phat_, axis=1)

            loss = self.compiled_loss(
                y,
                y_,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, phat_)

        return {m.name: m.result() for m in self.metrics}
    
    @splitter
    def test_step(self, x, y, sample_weight):
        with tf.GradientTape() as tape:
            vec = self(x, training=True)
            mean_, std_, shape_l_, shape_r_ = tf.unstack(vec, num=4, axis=-1)
        
            phat_ = Phat4TF(mean_, std_, shape_l_, shape_r_)
            y_ = NLL.calc(y, phat_, axis=1)

            loss = self.compiled_loss(
                y,
                y_,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )
        self.compiled_metrics.update_state(y, phat_)

        return {m.name: m.result() for m in self.metrics}

    def early_stop(self,
            monitor='val_loss', 
            min_delta=0.00001,
            patience=5, 
            verbose=1, 
            mode='min', 
            *args, **kwargs
        ):
        self._early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, 
            min_delta=min_delta, 
            patience=patience,
            verbose=verbose,
            mode=mode,
            *args, **kwargs
        )
    
    def fit(self, 
            *args, 
            early_stop:bool=True,
            **kwargs
        ):
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = []
        
        if not hasattr(self, '_early_stop') and early_stop:
            self.early_stop()
        
        if hasattr(self, '_early_stop'):
            kwargs['callbacks'].append(self._early_stop)

        return super().fit(*args, **kwargs)

class WeightsCallBack(tf.keras.callbacks.Callback):
    def __init__(self, weights={}):
        super(WeightsCallBack, self).__init__()  
        self.weights = weights
        
    def on_batch_end(self, batch, logs={}):
        curr_epoch = list(self.weights.keys())[-1]
        self.weights[curr_epoch].update({batch: self.model.get_weights()})

    def on_epoch_begin(self, epoch, logs={}):
        self.weights[epoch] = {}

class PhatNet(tf.keras.Model, GraphicMixin):
    PARAM_NAMES = ['mean', 'sig', 'shape_l', 'shape_r']
    def __init__(self, neurons=1):
        tf.keras.backend.set_floatx('float64')

        super(PhatNet, self).__init__(name='PhatNet')
            
        self.h1 = tf.keras.layers.Dense(neurons, input_shape=(1,), name='h1')

        self.mu = tf.keras.layers.Dense(1, name='mu')
        self.sigma = tf.keras.layers.Dense(1, activation='softplus', name='sigma')
        self.shape_l = tf.keras.layers.Dense(1, activation='softplus', name='shape_left')
        self.shape_r = tf.keras.layers.Dense(1, activation='softplus', name='shape_right')        

        self.body = tf.keras.layers.Concatenate(name='body')
        self.tails = tf.keras.layers.Concatenate(name='tails')
                
    def call(self, x):
        x = self.h1(x)

        mu_v = self.mu(x)
        sigma_v = self.sigma(x)
        shape_l_v = self.shape_l(x)
        shape_r_v = self.shape_r(x)

        body = self.body([mu_v, sigma_v])
        tails = self.tails([shape_l_v, shape_r_v])
        return body, tails
    
    @splitter
    def train_step(self, x, y, sample_weight):            
        with tf.GradientTape() as tape:
            body_, shapes_ = self(x, training=True)
            mean_, std_ = tf.unstack(body_, num=2, axis=-1)
            phat_ = Phat4TF(mean_, std_, shapes_[:,0], shapes_[:,1])
            nll = NLL.calc(y, phat_, axis=1)
            y_ = tf.concat((tf.reshape(nll, (-1,1)), shapes_), axis=1)

            loss = self.compiled_loss(
                y,
                y_,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, phat_)

        return {m.name: m.result() for m in self.metrics}
    
    def predicted_params(self):
        predicted = self.predict([0])
        vals = np.concatenate(list(predicted)).flatten()
        dict = {name: val for name, val in zip(self.PARAM_NAMES, vals)}
        df = pd.DataFrame([dict], index=['']).T

        return df

    @splitter
    def test_step(self, x, y, sample_weight):
        body_, shapes_ = self(x, training=True)
        mean_, std_ = tf.unstack(body_, num=2, axis=-1)

        phat_ = Phat4TF(mean_, std_, shapes_[:,0], shapes_[:,1])
        nll = NLL.calc(y, phat_, axis=1)

        y_ = tf.concat((tf.reshape(nll, (-1,1)), shapes_), axis=1)

        loss = self.compiled_loss(
            y,
            y_,
            sample_weight=sample_weight,
            regularization_losses=self.losses,
        )

        self.compiled_metrics.update_state(y, phat_)
        return {m.name: m.result() for m in self.metrics}
    
    def early_stop(self,
            monitor='val_loss', 
            min_delta=0.0001,
            patience=5, 
            verbose=1, 
            mode='min', 
            *args, **kwargs
        ):
        self._early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, 
            min_delta=min_delta, 
            patience=patience,
            verbose=verbose,
            mode=mode,
            *args, **kwargs
        )
    
    def fit(self, 
            *args, 
            weights:dict=None, 
            logdir:str='',
            early_stop:bool=True,
            nan_stop:bool=False,
            **kwargs
        ):
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = []
        
        if not hasattr(self, '_early_stop') and early_stop:
            self.early_stop()
        
        if hasattr(self, '_early_stop'):
            kwargs['callbacks'].append(self._early_stop)

        if nan_stop:
            kwargs['callbacks'].append(tf.keras.callbacks.TerminateOnNaN())

        if isinstance(weights, dict):
            kwargs['callbacks'].append(WeightsCallBack(weights))

        if logdir:
            kwargs['callbacks'].append(tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1))

        return super().fit(*args, **kwargs)
