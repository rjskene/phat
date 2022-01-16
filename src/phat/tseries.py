"""
Custom functionality for generating ARMA-GARCH time-series forecasts.
"""
from __future__ import annotations

from typing import Iterable, Union, Tuple, TYPE_CHECKING
import warnings
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

import numba as nb

from phat.utils import argsetter, arrayarize, PriceSim
from phat.dists import Phat

import pmdarima
import arch

ALLOWED_DISTS = Union[Phat, scipy.stats._distn_infrastructure.rv_frozen]

###### Numba helper functions to improve GarchForecast speed ######
@nb.njit(parallel=True, cache=True)
def random_normal(
    iters:int=1, 
    periods:int=252
    ) -> np.ndarray:
    """
    Loop creates 2d standard normal residual terms. Numba is much faster
    producing random 2d arrays in this fashion than via the `size` parameter

    Params
    -------
    iters:      int; number of separate simulations to run
    periods:    int; number of periods over which to run simulation

    Returns
    --------
    rvs:        np.ndarray of random draws from standard normal distribtion
    """
    rvs = np.zeros(shape=(iters, periods), dtype=np.float64)
    for i in nb.prange(iters):
        for j in nb.prange(periods):
            rvs[i, j] = np.random.normal()

    return rvs

@nb.njit(cache=True)
def _step_sum(
    params:Iterable[float], 
    values:Iterable[float]
    ) -> float:
    """
    Summation term of an AR(m), MA(n), or GARCH(p) / GARCH(q) process

    Each order-specific, fitted relationship parameter is multiplied by the
    corresponding order-specific lagged value, then summed across all the orders
    for that process.

    Params
    -------
    params:     np.ndarray; fitted parameters corresponding to the process
    values:     np.ndarray; lagged value against which the parameter is multiplied

    """
    sums = np.zeros(values.shape)
    for i in range(values.shape[0]):
        sums[i] = params[i]*values[i]

    return np.sum(sums)    

@nb.njit(cache=True)
def _arma_one_step(
    y:np.ndarray, 
    resids:np.ndarray, 
    constant:float, 
    ar_params:np.ndarray, 
    ma_params:np.ndarray, 
    m:int, 
    n:int
    ) -> float:
    """
    ARMA calculation for single step of forecast. Captures correlation of 
    current period mean return with prior period error terms.

    Params
    -------
    y:              np.ndarray; historical standardized residuals of ARMA-GARCH fit up to `maxlag` periods ago
    resids:         np.ndarray; historical residuals of ARMA-GARCH fit up to `maxlag` periods ago
    constant:       float; constant of the mean process4
    ar_params:      np.ndarray; paramters of the AR process
    ma_params:      np.ndarray; paramters of the MA process
    m:              int; order of the AR process
    n:              int; order of the MA process
    """
    ar = _step_sum(ar_params, y[-m:]) if m != 0 else 0
    ma = _step_sum(ma_params, resids[-n:]) if n != 0 else 0

    return constant + ar + ma

@nb.njit(cache=True)
def _garch_one_step(
    vols:np.ndarray,
    resids:np.ndarray,
    innov:np.ndarray,
    constant:float,
    arch_params:np.ndarray,
    garch_params:np.ndarray,
    p:int, 
    q:int
    ) -> float:
    """
    GARCH calculation for single step of forecast. Captures correlation of
    current period volatility with prior period residuals. The ARCH process is 
    analogous to the MA process for the mean. It is also a subset of GARCH, as:

                                ARCH(q) = GARCH(0,q)

    Params
    -------
    vols:           np.ndarray; historical volatities of ARMA-GARCH fit up to `maxlag` periods ago
    resids:         np.ndarray; historical residuals of ARMA-GARCH fit up to `maxlag` periods ago
    innov:          np.ndarray; random error terms
    constant:       float; constant of the vol process

    arch_params:      np.ndarray; paramters of the ARCH process
    garch_params:      np.ndarray; paramters of the GARCH process
    p:              int; order of the ARCH process
    q:              int; order of the GARCH process
                
    GARCH calculation for single step of forecast. Captures correlation of
    current period volatility with prior volatility. The "generalized" portion of
    "G"ARCH is analogous to the AR process for the mean. Unlike AR, however, it is
    uncommon for a volatility memory to be utilized without residual memory. So,
    you should always pair a `p=1` process with a `q=1` process.

    Remember for GARCH we work with volatilities so we must square them and the residuals.
    """
    arch = _step_sum(arch_params, resids[-q:]**2) if q != 0 else 0
    garch = _step_sum(garch_params, vols[-p:]**2) if p != 0 else 0

    var_t = constant + arch + garch
    resid_t = np.sqrt(var_t)*innov

    return var_t, resid_t
                   
class PROCESSMIXIN:
    """
    Mixin to provide common functionality to MEAN and VOL classes
    """
    @property
    def constant(self) -> float:
        return self.params[0]

    @property
    def max_order(self) -> int:
        return max(self.order)

    @property
    def props(self) -> Union[None, tuple]:
        return self.properties

    def _infer_order(self, params) -> tuple:
        """
        Infers the order of an ARMA or GARCH process.

        If params is even, assume even split between the params. If odd, assumes modulus lag belogs to first order.

        Parameters
        -----------
        params:     iterable; params of process from which order is inferred

        Return
        -------
        order:      tuple; 2x1 tuple with ARMA order (p,q) or GARCH order (m,n)
        """
        if any([params is None, isinstance(params, pd.Series) and params.empty, not params.any()]):
            first_order = second_order = 0
        else:
            first_order = len(params) // 2 + len(params) % 2
            second_order = len(params) // 2
        
        return first_order, second_order

class MEAN(PROCESSMIXIN):
    """
    Containter housing properties of the mean process.

    Typically, ARIMA or constant.

    Parameters
    -----------
    params:     iterable; values of constant plus any m, n process
    order:      tuple; 2x1 of m, n order (m = AR process, n = MA process)
    """
    def __init__(self,
        params:Iterable,
        order:tuple=None,
        ):
        self.params = params.values if hasattr(params, 'values') else params
        self._m, self._n = self._infer_order(params[1:]) if order is None else order
    
    @property
    def m(self) -> int:
        return self._m
    
    @property
    def n(self) -> int:
        return self._n
    
    @property
    def order(self) -> tuple:
        return self.m, self.n
    
    @property
    def ar_params(self) -> Iterable:
        return self.params[1:1+self.m]
    
    @property
    def ma_params(self) -> Iterable:
        return self.params[1+self.m:1+self.m+self.n]
    
    @property
    def properties(self) -> Union[None, tuple]:
        return self.constant, self.ar_params, self.ma_params, self.m, self.n

class VOL(PROCESSMIXIN):
    """
    Containter housing properties of the volatility process.

    Typically, constant, GARCH or some derivative of GARCH.

    Parameters
    -----------
    params:     iterable; values of constant plus any m, n process
    order:      tuple; 2x1 of p, q order
                where:  p = autoregressive to prior volatility 
                        n = autoregressive to prior residuals
    """    
    def __init__(self,
        params:Iterable,
        order:tuple=None
        ):
        self.params = params.values if hasattr(params, 'values') else params
        self._p, self._q = self._infer_order(params[1:]) if order is None else order
    
    @property
    def p(self) -> int:
        return self._p
    
    @property
    def q(self) -> int:
        return self._q
    
    @property
    def order(self) -> tuple:
        return self.p, self.q

    @property
    def arch_params(self):
        return self.params[1:1+self.q]
    
    @property
    def garch_params(self):
        return self.params[1+self.q:1+self.q+self.p]

    @property
    def properties(self) -> Union[None, tuple]:
        return self.constant, self.arch_params, self.garch_params, self.p, self.q

class GarchcastResults:
    """
    Container for outputs from Garchcast.forecast

    `plot` method provides several options for plotting results

    Parameters
    -----------
    values:     np.ndarray; 2d iters x periods array of values from Garchcast projection
    vols:       np.ndarray; 2d iters x periods array of forecast volatilities
    resids:     np.ndarray; 2d iters x periods array of forecast residuals relative to ARMA-GARCH
    """
    def __init__(self, 
        values:np.ndarray, 
        vols:np.ndarray, 
        resids:np.ndarray
        ):
        self.values = values
        self.vols = vols
        self.resids = resids

        self.iters, self.periods = values.shape

    @property
    def rets(self) -> np.ndarray:
        return 1 + self.values/100

    @property
    def vars(self) -> np.ndarray:
        return self.vols**2

    @property
    def sim(self) -> np.ndarray:
        if hasattr(self, '_sim'):
            return self._sim
        else:
            raise ValueError('You must plot a price simulation or call `_make_sim` directly')

    @argsetter(['iters', 'periods'])
    def _make_sim(self, iters:int, periods:int, p:float=1):
        """
        Creates a price simulator used to generate images
        """
        self._sim = PriceSim(p0=p, periods=periods, n=iters)
        return self._sim

    def plot(
        self, 
        kind:str='vol', 
        ax:Union[plt.Axes, Iterable[plt.Axes]]=None, 
        *args, **kwargs
        ) -> Union[plt.Axes, Iterable[plt.Axes]]:
        """
        Main method used to generate plots

        Params
        -------
        kind:    str; kind of plot. Accepts 'vol', 'var', 'price', or 'end_price'
        ax:      matplotlib Axes object(s); if one is not provided, it will be created.

        Return
        -------
        ax:      matplotlib Axes object(s)
        """
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

    def _plot_vol(self, ax:plt.Axes, *args, **kwargs) -> Union[plt.Axes, Iterable[plt.Axes]]:
        ax.plot((self.vols).mean(axis=0))
        ax.set_xlabel('Days')
        plt.suptitle('Phat-GARCH Forecast: Conditional Volatility')

        return ax
        
    def _plot_var(self, ax, *args, **kwargs) -> Union[plt.Axes, Iterable[plt.Axes]]:
        ax.plot((self.vols**2).mean(axis=0), *args, **kwargs)
        ax.set_xlabel('Days')
        plt.suptitle('Phat-GARCH Forecast: Conditional Variance')
        
        return ax

    def _plot_price(self, 
        p:float, 
        n:int, 
        axes:Iterable[plt.Axes], 
        *args, **kwargs
        ) -> Iterable[plt.Axes]:
        sim = self._make_sim(p=p)
        
        all_axes = []
        for i in np.random.randint(0, self.iters, size=n):
            _, __, chart_axes = sim.sim(rets=(1 + self.values[i]/100), axes=axes, show_chart=True, *args, **kwargs)
            all_axes.append(chart_axes)
        
        if axes is None:
            return np.array(all_axes)
        else:
            return axes

    def _plot_end_price(self, 
        p:float, 
        ax:plt.Axes, 
        *args, **kwargs
        ) -> Tuple[plt.Axes, np.ndarray, np.ndarray]:
        sim = self._make_sim(p=p)
        _, P, (ax, bins) = sim.sims(rets=1 + self.values/100, ax=ax, show_chart=True, *args, **kwargs)
        return ax, P, bins

class Garchcaster:
    """
Class for forecasting time series via ARMA-GARCH process.

Instances of a `pmdarima`-based ARIMA object and/or an ARCHModelResult from the 
`arch` package can be provided directly. Alternatively, the user can provide all
residuals, standardized residuals, conditional volatilities, the parameters for 
the ARMA-GARCH process, and the order.

Params:
    arma: instance of pmdarima ARIMA class OR iterable of floats 
        Either provide a pmdarima instance or the params derived from any arima process.
        If None, will search for AR process in `garch`, otherwise assumes ConstantMean.
    garch: instance of arch ARCHModelResult class OR iterable of floats
        either provide an ARCHModelResult instance or the params derived from any garch process
    y:  float or iterable of floats 
        iters x periods standardized residuals after filtering an ARMA-GARCH. Only required if
        garch is not ARCHModelResult type.
    vols: float or iterable of floats
        iters x periods conditional volatilities resulting from an ARMA-GARCH process. Only required if 
        garch is not ARCHModelResult type.
    resids: float or iterable of floats
        iters x periods residuals after filtering an ARMA-GARCH. Only required if garch is not ARCHModelResult type.
    iters: int 
        number of iterations of forecast
    periods: int 
        number of periods in each iteration of the forecast
    dist: Phat or scipy rv_frozen class 
        distribution used to generate residuals
    use_backcast: bool 
        use procedure to smooth values. See `here <https://arch.readthedocs.io/en/latest/univariate/generated/generated/arch.univariate.GARCH.backcast.html>`_.
    """
    GARCH_HAS_MEAN_PROCESS = False

    def __init__(self,
        garch:Union[arch.univariate.base.ARCHModelResult, Iterable[float]]=None,
        arma:Union[pmdarima.arima.arima.ARIMA, Iterable[float]]=None,
        y:Iterable[float]=None,
        vols:Iterable[float]=None,
        resids:Iterable[float]=None,
        iters:int=1, 
        periods:int=252,
        order:tuple=None,
        dist:ALLOWED_DISTS=None,
        use_backcast:bool=True,
        ):
        ### First, set self.mean
        if isinstance(arma, pmdarima.arima.arima.ARIMA):
            m, _, n = arma.order
            mean_order = (m,n)
            mean_params = arma.params()[:5]
        elif isinstance(garch, arch.univariate.base.ARCHModelResult):
            if isinstance(garch.model, arch.univariate.mean.ConstantMean):
                mean_order = (0, 0)
                mean_params = garch.params[:1]
            else:
                mean_order = (garch.model.lags[0].size, 0)
                mean_params = garch.params[:mean_order[0]+1]
                self.GARCH_HAS_MEAN_PROCESS = True
        elif arma is not None:
            mean_order = list(order)[:2] if order is not None else None
            mean_params = arrayarize(arma)
        
        self.mean = MEAN(mean_params, order=mean_order)

        ### Second, set self.vol
        if isinstance(garch, arch.univariate.base.ARCHModelResult):
            if hasattr(garch.model.volatility, 'p'):
                vol_order = (garch.model.volatility.p, garch.model.volatility.q)
            else:
                vol_order = (0, 0)

            vol_params = garch.params[mean_order[0]+1:] if self.GARCH_HAS_MEAN_PROCESS else garch.params[1:]
        elif isinstance(garch, (list, tuple, set, np.ndarray)):
            vol_params = arrayarize(garch)
            vol_order = list(order)[2:] if order is not None else None
        else:
            vol_order = (0, 0)

        self.vol = VOL(vol_params, order=vol_order)
        self.order = (*self.mean.order, *self.vol.order)

        if order is not None and order != self.order:
            msg = 'The `order` parameter provided does not match the order'
            msg += ' inferred from the models provided and has been OVERRIDDEN. The `order` parameter'
            msg += ' is not required when passing `arch` and/or pmdarima models to Garchcaster.'
            warnings.warn(msg)

        self.maxlag = max(self.order)

        ### Third, assign values
        if isinstance(garch, arch.univariate.base.ARCHModelResult):
            if any([y is not None, vols is not None, resids is not None]):
                raise ValueError('If `garch` is ARCHModelResult do not provide `y`, `vol`, or `resids`')
            else:
                y = garch.std_resid.values[self.maxlag:].copy()
                vols = garch.conditional_volatility.values[self.maxlag:].copy()
                resids = garch.resid.values[self.maxlag:].copy()
        else:
            y = arrayarize(y).copy()
            vols = arrayarize(vols).copy()
            resids = arrayarize(resids).copy()
        
        self.periods = periods
        self.iters = iters
   
        self.y_hist = y[-self.maxlag:]
        self.vols_hist = vols[-self.maxlag:]
        self.resids_hist = self._backcast(resids, self.maxlag) if use_backcast else resids[:-self.maxlag:]
        
        self._dist = dist
    
    @property
    def dist(self) -> ALLOWED_DISTS:
        if self._dist is None:
            NotImplementedError('You must pass `dist` parameter')
        else:
            return self._dist

    @argsetter(['iters', 'periods'], flat=True)
    def _sample_innovs(self, 
        iters:int=None, 
        periods:int=None, 
        dist:ALLOWED_DISTS=None,
        seed:int=None,
        ) -> np.ndarray:
        """
        Generates innovations according to the provided distribution. If a distribution is not
        provided, default is Gaussian.

        Params
        -------
        iters:          int; number of iterations of forecast
        periods:        int; number of periods in each iteration of the forecast
        dist:           Phat or scipy rv_frozen class; distribution used to generate residuals
        seed:           int; random state for innovation determination if `innovs` is not provided


        Return
        -------
        innovs:         np.ndarray
        """
        if self._dist is None and dist is None:
            innovs = random_normal(iters, periods)
        else:
            dist = dist if dist is not None else self._dist
            rvfunc = dist.std_rvs if hasattr(dist, 'std_rvs') else dist.rvs
            innovs = rvfunc(size=(iters, periods), seed=seed)

        return innovs

    def _backcast(self, resids, maxlag):
        power = 2
        tau = min(75, resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        backcast = np.sum((abs(resids[:tau]) ** power) * w)

        return np.repeat(backcast, maxlag)

    @staticmethod
    @nb.njit(cache=True)
    def _forecast_looper(
        y:np.ndarray,
        vols:np.ndarray,
        resids:np.ndarray,
        innovs:np.ndarray,
        iters:int, 
        periods:int,
        mean_args:tuple,
        vol_args:tuple,
        ) -> Tuple[np.ndarray]:
        """
        Main iterative, feed-forward function for forecasting an ARMA-GARCH process.
        
        Process
        --------
        To forecast a single-step period within a single iteration of the forecast

        1. First find the current variance. This requires historical variances for the
        p-requested lags.
        2. Find the residual term, which is a function of the volatility and the
        error term provided (or calculated as Standard Normal)
        3. Find mean value from ARMA
        4. Reassign y, errs, vols to feed forward into next iteration

        Params
        -------
        y:              np.ndarray; historical standardized residuals of ARMA-GARCH fit up to `maxlag` periods ago
        vols:           np.ndarray; historical volatilities of ARMA-GARCH fit up to `maxlag` periods ago
        resids:         np.ndarray; historical residuals of ARMA-GARCH fit up to `maxlag` periods ago
        innovs:         iters x periods np.ndarray of garch residuals
        iters:          int; number of iterations of forecast
        periods:        int; number of periods in each iteration of the forecast
        dist:           Phat or scipy rv_frozen class; distribution used to generate residuals
        mean_args:      tuple; properties of the mean process use in `_arma_one_step`    
        vol_args:       tuple; properties of the vol process use in `_garch_one_step`
        """
        y_ = np.zeros((iters, periods))
        vols_ = np.zeros((iters, periods))
        resids_ = np.zeros((iters, periods))
        for i in range(iters):
            y_i = y.copy()
            vols_i = vols.copy()
            resids_i = resids.copy()
            for t in range(periods):
                mu_t = _arma_one_step(y_i, resids_i, *mean_args)
                var_t, resid_t = _garch_one_step(vols_i, resids_i, innovs[i, t], *vol_args)

                y_[i, t] = mu_t + resid_t
                vols_[i, t] = np.sqrt(var_t)
                resids_[i, t] = resid_t
                
                y_i[-1] = y_[i, t]
                vols_i[-1] = vols_[i, t]
                resids_i[-1] = resids_[i, t]                
            
                if y.size > 1:
                    y_i[-y_i.size:-1] = y_i[-y_i.size+1:]
                    vols_i[-y_i.size:-1] = vols_i[-y_i.size+1:]
                    resids_i[-y_i.size:-1] = resids_i[-y_i.size+1:]

                # if np.isnan(y_).any() or np.isnan(y).any():
                #     print (i, t, mu_t, resid_t, innovs[i,t], y, vols, resids)
        return y_, vols_, resids_

    @argsetter(['dist', 'iters', 'periods'], flat=True)
    def forecast(self, 
        iters:int=None, 
        periods:int=None, 
        dist:ALLOWED_DISTS=None,
        innovs:Iterable=None,
        seed:int=None,
        ) -> GarchcastResults:
        """
        User-facing interface with forecast generator

        Params:
            iters: int
                number of iterations of forecast
            periods: int
                number of periods in each iteration of the forecast
            dist: Phat or scipy rv_frozen class
                distribution used to generate residuals
            innovs: np.ndarray
                iters x periods of garch residuals
            seed: int
                random state for innovation determination if `innovs` is not provided
        """
        if innovs is None:
            innovs = self._sample_innovs(iters, periods, dist=dist, seed=seed)

        results = Garchcaster._forecast_looper(
            self.y_hist, 
            self.vols_hist, 
            self.resids_hist, 
            innovs,
            iters, periods,
            self.mean.props,
            self.vol.props
        )
        return GarchcastResults(*results)
