"""
Collection of helper functions
"""

import numpy as np
import pandas as pd
from functools import wraps
from typing import Union, Iterable, Callable, Sequence

import matplotlib.pyplot as plt

XTYPE = Union[float, Iterable[Sequence[float]]]

def cosh_squared(x,k):
    return .5*(np.cosh(2*k*x) + 1)

def sech_squared(x,k):
    return 1 / cosh_squared(x, k)

def arrayarize(val:Iterable):
    """
    Converts list-like objects to np.ndarray
    """
    list_types = (list, tuple, set, pd.Series)
    if isinstance(val, np.ndarray):
        pass
    elif isinstance(val, list_types):
        val = np.array(val)
    elif isinstance(val, (int, float)):
        val = np.array([val])
    else:
        text = 'You must provide an iterable of type: '
        text += ', '.join(list_types)
        raise ValueError(text)
    return val

def argsetter(kws:Union[Iterable, str]='x', flat=False):
    if isinstance(kws, str):
        kws = list(kws)

    def deco(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if args:
                for k, arg in zip(kws, args):
                    kwargs[k] = arg if flat else arrayarize(arg)
            for k in kws:
                if k not in kwargs or kwargs[k] is None:                
                    kwargs[k] = getattr(self, k) if flat else arrayarize(getattr(self, k))
            
            return func(self, **kwargs)
    
        return wrapper

    return deco

def dotweight(func) -> Callable:
    """
    Takes the weighted average of values in the array `stack`
    Weights provided by array `p`. average found via dot product.

    Used as a wrapper for class methods: `p` must be an attribute of the class instance

    Returns
    --------
    np.ndarray; max 3 dimensions, the weighted averages along one dimension of `stack`
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> np.ndarray:
        stack = func(self, *args, **kwargs)
        if stack.ndim == 3:
            p = np.tile(self.p, (stack.shape[0],stack.shape[1],1))
            return np.multiply(p, stack).sum(axis=2)
        else:
            return (self.p @ stack)

    return wrapper

def stacker(arr):
    """
    arr is numpy array
    """
    arr = arrayarize(arr)
    if arr.ndim == 1:
        return np.vstack
    elif arr.ndim == 2:
        return np.dstack
    else:
        raise ValueError(f'`arr` has {arr.ndim}')

class PriceSim:
    """
    Handler for generating price time-series from a set of return time-series
    """
    def __init__(self, p0:float, rets=None, periods:int=0, n:int=0):
        self.p0 = p0
        self.rets = rets
        self.n = n
        self.periods = periods
    
    @argsetter(['p0', 'rets', 'periods'])
    def sim(self, p0=None, rets=None, periods=None, show_chart:bool=False, *args, **kwargs):
        S = p0*rets.cumprod()
        if show_chart:
            return rets, S, self.sim_chart(rets, S, periods, *args, **kwargs)
        else:
            return rets, S

    @argsetter(['p0', 'rets'], flat=True)
    def sims(self, 
        p0:float=None, 
        rets=None, 
        show_chart:bool=False, 
        *args, **kwargs
    ):
        S = p0*rets.cumprod(axis=1)
        
        if show_chart:
            return rets, S, self.sims_chart(S, *args, **kwargs)
        else:
            return rets, S
        
    def sim_chart(self, rets, S, periods, axes:Iterable=None, title=''):
        if axes is None:
            fig, axes = plt.subplots(1,2,figsize=(14,5))
        elif len(axes) != 2:
            raise ValueError('Chart requires two subplots')
        
        ax1, ax2 = axes
        
        x = np.arange(periods)
        ax1.plot(x, rets - 1)
        ax2.plot(x, S)

        ax1.set_title('Daily Returns')
        ax2.set_title('Asset Price')

        if not title:
            title = 'Phat Price Simulation'
            
        plt.suptitle(title, y=1)
        
        return axes
    
    def sims_chart(self, S, ax=None, title='', *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(10,6))

        if 'bins' in kwargs:
            if isinstance(kwargs['bins'], (float, int)):
                bins = np.linspace(0, S[:, -1].max(), kwargs['bins'])
            else:
                bins = kwargs['bins']
            kwargs.pop('bins')
        else:
            bins = np.linspace(0, S[:, -1].max(), 250)
                                
        counts, bins, _ = ax.hist(S[:, -1], bins=bins, density=True, *args, **kwargs)

        if not title:
            title = 'Distribution of Ending Share Prices'
        
        ax.set_title(title)

        return ax, bins
