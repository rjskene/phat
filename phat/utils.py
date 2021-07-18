import numpy as np
from functools import wraps
from typing import Union, Iterable

import matplotlib.pyplot as plt

def make_arr(v):
    if isinstance(v, (float, int)):
        v = np.array([v])
    if isinstance(v, (list, tuple, set)):
        v = np.array(v)
    return v

def argsetter(kws:Union[Iterable, str]='x', flat=True):
    if isinstance(kws, str):
        kws = list(kws)

    def deco(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if args:
                for k, arg in zip(kws, args):
                    kwargs[k] = arg if flat else make_arr(arg)
            for k in kws:
                if k not in kwargs or kwargs[k] is None:                
                    kwargs[k] = getattr(self, k) if flat else make_arr(getattr(self, k))
            
            return func(self, **kwargs)
    
        return wrapper

    return deco

class PriceSim:
    def __init__(self, p0:float, rets=None, days:int=0, n:int=0):
        self.p0 = p0
        self.rets = rets
        self.n = n
        self.days = days
    
    @argsetter(['p0', 'rets', 'days'])
    def sim(self, p0=None, rets=None, days=None, show_chart:bool=False, *args, **kwargs):
        S = p0*rets.cumprod()
        if show_chart:
            return rets, S, self.sim_chart(rets, S, days)
        else:
            return rets, S

    @argsetter(['p0', 'rets', 'days', 'n'])
    def sims(self, p0:float=None, rets=None, days:int=None, n:int=None, show_chart:bool=False, *args, **kwargs):        
        S = p0*rets.cumprod(axis=1)
        
        if show_chart:
            return rets, S, self.sims_chart(S, *args, **kwargs)
        else:
            return rets, S
        
    def sim_chart(self, rets, S, days, axes:Iterable=None, title=''):
        if axes is None:
            fig, axes = plt.subplots(1,2,figsize=(14,5))
        elif len(axes) != 2:
            raise ValueError('Chart requires two subplots')

        ax1, ax2 = axes
        
        x = np.arange(days)
        ax1.plot(x, rets - 1)
        ax2.plot(x, S)

        ax1.set_title('Daily Returns')
        ax2.set_title('Share Price')

        if not title:
            title = 'Phat Share Price Simulation'
            
        plt.suptitle(title, y=1.05)
        
        return axes
    
    def sims_chart(self, S, ax=None, title='', *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(10,6))

        if 'bins' in kwargs:
            bins = np.linspace(0, S[:, -1].max(), kwargs['bins'])
            kwargs.pop('bins')
        else:
            bins = np.linspace(0, S[:, -1].max(), 250)
                                
        counts, bins, _ = ax.hist(S[:, -1], bins=bins, density=True, *args, **kwargs)

        if not title:
            title = 'Distribution of Ending Share Prices'
        ax.set_title(title)

        return ax, bins