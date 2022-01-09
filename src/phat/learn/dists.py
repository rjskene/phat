"""
Tensorflow does not support scipy, therefore, all distributions in 
phat/dists.py must be replicated via tensorflow_probability 
"""

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from tensorflow_probability.python.internal import reparameterization, dtype_util, \
    tensor_util, parameter_properties
from tensorflow_probability.python.bijectors import softplus as softplus_bijector, identity as identity_bijector

### DEBUG UTILS ###
@tf.function
def find_nans(val):
    cond = tf.math.is_nan(val)
    idx = tf.where(cond)
    return idx

@tf.function
def find_zeros(val):
    cond = tf.math.equal(val, 0)
    idx = tf.where(cond)
    return idx

@tf.function
def if_not_empty(x, idx):
    if tf.not_equal(tf.size(idx), 0):
        return x[idx[0,0]]
    else:
        return tf.constant(0., dtype=np.float64)

# idx = find_nans(####)
# tf.print('loc', self.loc[0], self.scale[0], self.mean[0], self.std[0], self.shape[0])
# tf.print('idx', if_not_empty(x, idx), -if_not_empty(x, idx) + 10**-1, idx)

def gptf_prob(
    gptf,
    x,
    model
    ):
    loc = tf.convert_to_tensor(gptf.loc)
    scale = tf.convert_to_tensor(gptf.scale)
    concentration = tf.convert_to_tensor(gptf.concentration)
    conc_zero = tf.equal(concentration, 0)
    conc = tf.where(conc_zero, tf.constant(1, gptf.dtype), concentration)
    conc_neg = tf.math.less(conc, 0)
    neg_max = loc - (scale / conc)

    z = gptf._z(x, scale)
    base = tf.ones_like(z, gptf.dtype) + z*conc
    pow_ = -1 / conc - tf.ones_like(z, gptf.dtype)
    # tf.print('\n****RIGHT TAIL???', model.rtail)
    # tf.print('\nXXXXXXX', x)
    # tf.print('\nLOC', loc)
    # tf.print('\nSCALE', scale)
    # tf.print('\nZZZZZZZ', z)
    # tf.print('\nCONC', conc)
    # tf.print('\nBASE', base)
    # tf.print('POW_', pow_)
    
    base_pow_ = tf.math.pow(base, pow_) / gptf.scale

    # idx3 = find_nans(base_pow_)
    # tf.print('BASE_POW_', if_not_empty(base_pow_, idx3), idx3)
    # tf.print(base_pow_)
    
    # if tf.not_equal(tf.size(idx3), 0):
    #     tf.print('RTail?', model.rtail)
    #     tf.print('BASE_POW_', tf.gather_nd(base_pow_, idx3))
    #     tf.print (tf.shape(conc), tf.shape(idx3))
        
    #     tf.print('X', tf.shape(x), x[idx3[0,0]])
        
    #     tf.print('Loc', loc[idx3[0,0]], tf.unique(loc))
    #     tf.print(x<loc[0])
    #     tf.print('X<loc', tf.shape(x[x<loc[0]]))
    #     tf.print('Scale', tf.shape(gptf.scale), tf.unique(gptf.scale))
    #     tf.print('Z', tf.shape(z), tf.unique(z[idx3[0,0]]))
    #     tf.print('CONC_', conc[idx3[0,0]])
    #     tf.print('BASE', tf.shape(base[idx3[0,0]]), tf.unique(base[idx3[0,0]]))
        
    #     inval_conc = conc[idx3[0,0]]
    #     inval_z = z[idx3[0,0]][0]
    #     inval_scale = gptf.scale[0]
    #     inval_base = 1 + inval_z*inval_conc
    #     inval_pow = -1 / inval_conc - 1
        
    #     tf.print('inval conc', inval_conc, 'inval base', inval_base, 'inval pow', inval_pow)
    #     inval_base_pow = tf.math.pow(inval_base, inval_pow) / inval_scale
    #     tf.print('inval base pow, PRE SCALE', tf.math.pow(inval_base, inval_pow))
    #     tf.print('inval base pow', inval_base_pow)

    # def base_pow_(x, scale, conc, gptf):
        # return base_pow_

    # valid_x = tf.greater_equal(x, loc[0])
    # w_valid_x = tf.where(valid_x, calc_prob(x), tf.constant(0, gptf.dtype))

    w_negconc = tf.where((x>=loc) & (x<=neg_max), base_pow_, 0+10**-10)
    w_posconc = tf.where(x >= loc, base_pow_, 0+10**-10)
    
    return tf.where(
        conc_zero, z, tf.where(conc_neg, w_negconc, w_posconc)
    )

class CarbenBase4TF(tfd.Distribution):    
    def __init__(self, mean, std, shape,
        validate_args=False, 
        allow_nan_stats=True,
        name='CarbenBase'
        ):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = tf.float64
            self._mean = tensor_util.convert_nonref_to_tensor(mean, dtype=dtype, name='mean')
            self._std = tensor_util.convert_nonref_to_tensor(std, dtype=dtype, name='std')
            self._shape = tensor_util.convert_nonref_to_tensor(shape, dtype=dtype, name='shape')
            self._loc = self._calc_loc()
            self._scale = self._calc_scale()
            self._body = tfd.Normal(self._mean, self._std)
            super(CarbenBase4TF, self).__init__(
              dtype=dtype,
              reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              parameters=parameters,
              name=name
            )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
        return dict(
            mean=parameter_properties.ParameterProperties(),
            std=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))
            ),
            shape=parameter_properties.ParameterProperties()
        )
        # pylint: enable=g-long-lambda
        
    @property
    def mean(self):
        return self._mean
    
    @property
    def std(self):
        return self._std
    
    @property
    def shape(self):
        return self._shape

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return self._scale

    @property
    def parameters(self):
        params = super().parameters
        params['loc'] = self.loc
        params['scale'] = self.scale
        return params

    def _z_for_W(self):
        """
        in Carreau (2008), z defined as:

            (1 + shape)**2 / 2*pi
        """
        num = (1 + self.shape)**2
        denom = 2*np.pi
        return  num / denom        

    def _W_z(self):
        """
        in Carreau (2008), W(z) can be calculated directly as:
            
            std**2 * (1 + shape)**2 / scale**2

        scale is not a free parameter, however, so we must calculate W(z) via
        the Lambert function on z, which is defined only in free parameters.
        """

        return tfp.math.lambertw(self._z_for_W())

    @property
    def gamma(self):
        val = tf.math.sqrt(self._W_z() / 2)
        return 1 + .5*(1 + tf.math.erf(val))
    
    def _calc_scale(self):
        num = self._std*(1 + self._shape)
        denom = tf.math.sqrt(self._W_z())
        return num / denom

    @property
    def body(self):
        if hasattr(self, '_body'):
            return self._body
        else:
            raise NotImplementedError

    @property
    def tail(self):
        if hasattr(self, '_tail'):
            return self._tail
        else:
            raise NotImplementedError

    def _log_prob(self, x):
        prob = self._prob(x)
        return tf.math.log(prob)

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])
    
    def _default_event_space_bijector(self):
        return identity_bijector.Identity(validate_args=self.validate_args)

class CarbenRight4TF(CarbenBase4TF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tail = tfd.GeneralizedPareto(loc=self._loc, scale=self._scale, concentration=self._shape)
    
    def _calc_loc(self):
        return self.mean + self.std*tf.math.sqrt(self._W_z())

    def _prob(self, x):
        """
        Right tail only.
        where x>a,
            f(x) = f_t(x) / gamma
        """
        x_tail = x > self.loc
        # tf.print('is _prob: any tail????', x_tail)
        # tf.print('Rtail?', self.rtail, 'does this work?', tf.shape(x[x>self.loc[0]]))
        prob = tf.where(
            x_tail,
            gptf_prob(self.tail, tf.where(x_tail, x, self.loc), self),
            self.body.prob(x) + 10**-10,
        )
        return prob / self.gamma

class CarbenLeft4TF(CarbenBase4TF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tail = tfd.GeneralizedPareto(loc=-self._loc, scale=self._scale, concentration=self._shape)
    
    def _calc_loc(self):
        return self.mean - self.std*tf.math.sqrt(self._W_z())

    def _prob(self, x):
        """
        Left tail only.
        where x>a,
            f(x) = f_t(x) / gamma
        """
        x_tail = x < self.loc
        # tf.print ('in _prob: right taill?????', self.rtail)
        # tf.print('in _prob: xxxxxx', x)
        # tf.print('in _prob: LOC', self.loc)
        # tf.print('is _prob: any tail????', x_tail)
        prob = tf.where(
            x < self.loc,
            gptf_prob(self.tail, tf.where(x_tail, -x, -self.loc), self),
            self.body.prob(x) + 10**-10,
        )
        return prob / self.gamma

class CarbenHybrid4TF:
    def __new__(cls, *args, **kwargs):
        args = list(args)
        shape_is_arg =  len(args) >= 2
        shape = args[2] if shape_is_arg else kwargs['shape']
        
        rtail_is_arg = len(args) >= 4
        if rtail_is_arg:
            rtail = args.pop(3)
        else:
            rtail = kwargs.pop('rtail') if 'rtail' in kwargs else True

        if isinstance(shape, float):
            if shape < 0 and not rtail:
                txt = 'If you provide a negative shape parameter,'
                txt += ' do not provide `rtail`'
                txt += ' as a left-tailed Carben is computed automatically.'
                raise ValueError(txt)             
            elif shape < 0:
                rtail = False
                if shape_is_arg:
                    args[2] = -shape
                else:
                    kwargs['shape'] = -shape
        
        if rtail:
            kwargs['name'] = 'CarbenRight'
            obj = CarbenRight4TF(*args, **kwargs)
            obj.rtail = True
        else:
            kwargs['name'] = 'CarbenLeft'            
            obj = CarbenLeft4TF(*args, **kwargs)
            obj.rtail = False
            
        return obj

class PhatMixture(tfd.Mixture):
    def __repr__(self):
        prepend = '<Phat4TF (inherits from tfp.distributions.Mixture) '
        append = ' '.join(super().__repr__().split(' ')[2:])
        return prepend + append

    def __str__(self):
        return str(self.__repr__())

class Phat4TF:
    def __new__(cls, mean, std, shape_l, shape_r, mix=.5):
        m = tf.shape(mean)[0]
        mix = tf.constant([mix, 1-mix], dtype=tf.float64)
        p = tf.ones([m, 1], dtype=tf.float64) * mix
        
        c1 = CarbenHybrid4TF(mean, std, shape_l, rtail=False)
        c2 = CarbenHybrid4TF(mean, std, shape_r)
        
        obj = PhatMixture(
            cat=tfd.Categorical(probs=p),
            components=[c1,c2]
        )
        obj.left = obj.components[0]
        obj.right = obj.components[1]
        return obj

