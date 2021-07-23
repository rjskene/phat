import functools
from typing import Iterable, Union
import numpy as np
import matplotlib.pyplot as plt

import numba as nb

from phat.utils import argsetter, arrayarize, PriceSim

@nb.njit(parallel=True)
def random_normal(iters=1, periods=252):
    """
    Loop for create 2d standard normal error terms. Numba is much faster
    producing random 2d arrays in this fashion than via the `size` parameter
    """
    rvs = np.zeros(shape=(iters, periods), dtype=np.float64)
    for i in nb.prange(iters):
        for j in nb.prange(periods):
            rvs[i, j] = np.random.normal()

    return rvs

@nb.njit
def _step_sum(params, values):
    if (len(params)) != len(values):
        raise ValueError('Number of params does not match number of values.')
    sums = np.zeros(values.shape)
    for i in nb.prange(values.shape[0]):
        sums[i] = params[i]*values[i]

    return np.sum(sums)    

@nb.njit
def _arma_one_step(y, errs, intercept, ar_params, ma_params, m, n):
    """
    Moving average calculation for single step of forecast. Captures correlation of
    current period mean return with prior period error terms

    ma_t = sum(theta_i*err_t-i for i in range(n))

    The order, n, is inferred from the length of values and params arguments provided.

    Params
    -------
    params:     (n x 1) np.ndarray;
                MA coefficients for MA fitted model

    values:     (n x 1) np.ndarray; 
                historical error terms correlated with current value per model
    """
    if m == 0:
        ar = 0
    else:
        ar = _step_sum(ar_params, y[-m:])
    if n == 0:
        ma = 0
    else:
        ma = _step_sum(ma_params, errs[-n:])

    return intercept + ar + ma

@nb.njit
def _garch_one_step(vols, resids, innov, mean, intercept, arch_params, garch_params, p, q):
    """
    ARCH calculation for single step of forecast. Captures correlation of
    current period volatility with prior period residuals. The ARCH process is 
    analogous to the MA process for the mean. It is also a subset of GARCH, as:

                                ARCH(q) = GARCH(0,q)

    arch_t = alpha_0 + sum(a_i*err_t-i for i in range(q))

    The order, q, is inferred from the length of values and params arguments provided.

    Params
    -------
    params:     (q x 1) np.ndarray;
                MA coefficients for MA fitted model

    values:     (q x 1) np.ndarray; 
                historical error terms correlated with current value per model
    GARCH calculation for single step of forecast. Captures correlation of
    current period volatility with prior volatility. The "generalized" portion of
    "G"ARCH is analogous to the AR process for the mean. Unlike AR, however, it is
    uncommon for a volatility memory to be utilized without residual memory. So,
    you should always pair a `p=1` process with a `q=1` process.

    garch_t = sum(beta_i*err_t-i**2 for i in range(p))

    FOR GARCH HAVE TO SQUARE THE RESIDUAL AND OBVIOUSLY THE VOLATILITY!!!!!!!!!!!

    The order, p, is inferred from the length of values and params arguments provided.

    Params
    -------
    params:     (p x 1) np.ndarray;
                MA coefficients for MA fitted model

    values:     (p x 1) np.ndarray; 
                historical error terms correlated with current value per model
    """
    if q == 0:
        arch = 0
    else:
        arch = _step_sum(arch_params, resids[-q:]**2)
    if p == 0:
        garch = 0
    else:
        garch =_step_sum(garch_params, vols[-p:]**2)

    var_t =  intercept + arch + garch
    resid_t = np.sqrt(var_t)*innov

    return var_t, resid_t

def orderfilter(func):
    @functools.wraps(func)
    def wrap(self, params):
        """
        If `orders=None`, infer lag numbers from params.
        """
        if params is None:
            first_order = second_order = 0
        else:
            first_order, second_order = func(self, params)
        
        return first_order, second_order

    return wrap
                   
class ORDERMIX:
    def _infer_order(self, params):
        """
         If params is even, assume even split
        between the params. If odd, assumes modulus lag belogs to first order.
        
        """
        first_order = len(params) // 2 + len(params) % 2
        second_order = len(params) // 2
        
        return first_order, second_order
        
class ARMA(ORDERMIX):
    def __init__(self,
        params=None,
        order=None,
        ):
        self.params = params
        self._m, self._n = self._infer_order(params) if order is None else order
    
    @orderfilter
    def _infer_order(self, params):
        params = params[1:]
        return super()._infer_order(params)
    
    @property
    def m(self):
        return self._m
    
    @property
    def n(self):
        return self._n
    
    @property
    def order(self):
        return self.m, self.n
    
    @property
    def intercept(self):
        return self.params[0]
    
    @property
    def ar_params(self):
        return self.params[1:1+self.m]
    
    @property
    def ma_params(self):
        return self.params[1+self.m:1+self.m+self.n]
    
    @property
    def arma_args(self):
        if self.m == 0 and self.n == 0:
            return None
        else:
            return self.intercept, self.ar_params, self.ma_params, self.m, self.n
    
class GARCH(ORDERMIX):
    def __init__(self,
        params=None,
        order=None):
        
        self.params = params
        self._p, self._q = self._infer_order(params) if order is None else order
    
    @orderfilter
    def _infer_order(self, params):
        params = params[2:]
        return super()._infer_order(params)
    
    @property
    def mean(self):
        return self.params[0]
    
    @property
    def p(self):
        return self._p
    
    @property
    def q(self):
        return self._q
    
    @property
    def order(self):
        return self.p, self.q
    
    @property
    def intercept(self):
        return self.params[1]
    
    @property
    def arch_params(self):
        return self.params[2:2+self.q]
    
    @property
    def garch_params(self):
        return self.params[2+self.q:2+self.q+self.p]
    
    @property
    def garch_args(self):
        if self.p == 0 and self.q == 0:
            return None
        else:
            return self.mean, self.intercept, self.arch_params, self.garch_params, self.p, self.q
    
class Garchcaster:
    def __init__(self, 
        y,
        vols,
        resids,
        arma=None,
        garch=None,
        iters=1, 
        periods=252,
        order=None,
        dist=None,
        use_backcast:bool=True
    ):
        y = arrayarize(y).copy()
        vols = arrayarize(vols).copy()
        resids = arrayarize(resids).copy()
        
        self.periods = periods
        self.iters = iters

        self.arma_params = arrayarize(arma) if arma is not None else arma
        arma_orders = list(order)[:2] if order else None
        self.arma = ARMA(self.arma_params, order=arma_orders)
        self.garch_params = arrayarize(garch)
        garch_orders = list(order)[2:] if order else None
        self.garch = GARCH(self.garch_params, order=garch_orders)
                        
        self.order = (*self.arma.order, *self.garch.order)
        self.maxlag = max(self.order)
        
        self.y_hist = y[-self.maxlag:]
        self.vols_hist = vols[-self.maxlag:]
        self.resids_hist = self._backcast(resids, self.maxlag) if use_backcast else resids[:-self.maxlag:]
        
        self._dist = dist
    
    @property
    def dist(self):
        if self._dist is None:
            NotImplementedError('You must pass `dist` parameter')
        else:
            return self._dist

    @staticmethod
    @nb.njit
    def _forecast_loop(
        y, vols, resids,
        innovs, iters, periods,
        arma_args:nb.optional(tuple),
        garch_args:nb.optional(tuple),        
        ):
        """
        For a complete single-step,        
        1. First find the current variance. This requires historical variances for the
        p-requested lags.
        2. Find the residual term, which is a function of the volatility and the
        error term provided (or calculated as Standard Normal)
        3. Find mean value from ARMA
        4. Reassign y, errs, vols to incorporate new values found in the current iteration
        """
        r_ = np.zeros((iters, periods))
        vols_ = np.zeros((iters, periods))
        resids_ = np.zeros((iters, periods))
        
        for i in nb.prange(iters):
            y_i = y.copy()
            vols_i = vols.copy()
            resids_i = resids.copy()
            for t in range(periods):
                if garch_args is None:
                    var_t = 1
                    resid_t = innovs[i, t]
                else:
                    var_t, resid_t = _garch_one_step(vols_i, resids_i, innovs[i, t], *garch_args)

                if arma_args is None:
                    mu_t = garch_args[0]
                else:
                    mu_t = _arma_one_step(y_i, resids_i, *arma_args)

                r_[i, t] = mu_t + resid_t
                vols_[i, t] = np.sqrt(var_t)
                resids_[i, t] = resid_t
                
                y_i[-1] = r_[i, t]
                vols_i[-1] = vols_[i, t]
                resids_i[-1] = resids_[i, t]                
            
                if y.size > 1:
                    y_i[-y_i.size:-1] = y_i[-y_i.size+1:]
                    vols_i[-y_i.size:-1] = vols_i[-y_i.size+1:]
                    resids_i[-y_i.size:-1] = resids_i[-y_i.size+1:]

                if np.isnan(y).any():
                    print (i, t, mu_t, resid_t, innovs[i,t], y, vols, resids)
        return r_, vols_, resids_

    @argsetter(['iters', 'periods'], flat=True)
    def _sample_innovs(self, iters, periods, dist=None):
        if self._dist is None and dist is None:
            innovs = random_normal(iters, periods)
        else:
            dist = dist if dist is not None else self._dist
            rvfunc = dist.std_rvs if hasattr(dist, 'std_rvs') else dist.rvs
            innovs = rvfunc(size=(iters, periods))

        return innovs

    def _backcast(self, resids, maxlag):
        power = 2
        tau = min(75, resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        backcast = np.sum((abs(resids[:tau]) ** power) * w)

        return np.repeat(backcast, maxlag)

    @argsetter(['dist', 'iters', 'periods'], flat=True)
    def forecast(self, iters:int=None, periods:int=None, dist=None, innovs:Iterable=None):

        if innovs is None:
            innovs = self._sample_innovs(iters, periods, dist=dist)
    
        results = Garchcaster._forecast_loop(
            self.y_hist, self.vols_hist, self.resids_hist, innovs,
            iters, periods,
            self.arma.arma_args, self.garch.garch_args
        )
        return GarchcastResults(*results)

class GarchcastResults:
    def __init__(self, values, vols, resids):
        self.values = values
        self.vols = vols
        self.resids = resids

        self.iters, self.periods = values.shape

    @property
    def vars(self):
        return self.vols**2

    @property
    def sim(self):
        if hasattr(self, '_sim'):
            return self._sim
        else:
            raise ValueError('You must plot a price simulation or call `_make_sim` directly')

    @argsetter(['iters', 'periods'])
    def _make_sim(self, iters:int, periods:int, p:float=1):
        self._sim = PriceSim(p0=p, periods=periods, n=iters)
        return self._sim

    def plot(self, kind='vol', ax:Union[plt.Axes, Iterable[plt.Axes]]=None, *args, **kwargs):
        multi_ax = ['price']
        if ax is None and kind not in multi_ax:
            fig, ax = plt.subplots(1,1,figsize=(10,6))

        if kind == 'vol':
            ax = self._plot_vol(ax, *args, **kwargs)
        elif kind == 'var':
            ax = self._plot_var(ax, *args, **kwargs)
        elif kind == 'price':
            if len(args) > 1:
                p = args[0]
            if len(args) > 2:
                n = args[2]
            if 'p' in kwargs:
                p = kwargs['p']
                kwargs.pop('p')
            if 'n' in kwargs:
                n = kwargs['n']
                kwargs.pop('n')
            
            ax = self._plot_price(p=p, n=n, axes=ax, *args, **kwargs)
        elif kind == 'end_price':
            if len(args) > 1:
                p = args[0]
            if 'p' in kwargs:
                p = kwargs['p']
                kwargs.pop('p')

            return self._plot_end_price(p=p, ax=ax, *args, **kwargs)
        else:
            raise ValueError(f'`kind:` {kind} not supported')

        return ax

    def _plot_vol(self, ax:plt.Axes, *args, **kwargs):
        ax.plot((self.vols).mean(axis=0))
        ax.set_xlabel('Days')
        plt.suptitle('Phat-GARCH Forecast: Conditional Volatility')

        return ax
        
    def _plot_var(self, ax, *args, **kwargs):
        ax.plot((self.vols**2).mean(axis=0), *args, **kwargs)
        ax.set_xlabel('Days')
        plt.suptitle('Phat-GARCH Forecast: Conditional Variance')
        
        return ax

    def _plot_price(self, p, n, axes:Iterable[plt.Axes], *args, **kwargs):
        sim = self._make_sim(p=p)
        
        all_axes = []
        for i in np.random.randint(0,self.iters, size=n):
            _, __, chart_axes = sim.sim(rets=(1 + self.values[i]/100), axes=axes, show_chart=True, *args, **kwargs)
            all_axes.append(chart_axes)
        
        if axes is None:
            return np.array(all_axes)
        else:
            return axes

    def _plot_end_price(self, p, ax:plt.Axes, *args, **kwargs):
        sim = self._make_sim(p=p)
        _, P, (ax, bins) = sim.sims(rets=1 + self.values/100, ax=ax, show_chart=True, *args, **kwargs)
        return ax, P, bins
