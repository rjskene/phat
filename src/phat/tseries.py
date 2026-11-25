"""
Custom functionality for generating ARMA-GARCH time-series forecasts.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import scipy.stats

if TYPE_CHECKING:  # optional packages, imported for annotations only
    import pmdarima
    from arch.univariate.base import ARCHModelResult

from phat._compat import renamed_attribute, renamed_kwargs
from phat.dists import Phat, _rng
from phat.utils import PriceSim, argsetter, arrayarize

# Optional accelerators and model packages. Everything below degrades gracefully:
# without Numba the forecast loop runs in pure Python; without `arch` only the
# array-based constructor is available.
try:
    import numba as nb
    _njit = nb.njit
except ImportError:  # pragma: no cover - exercised in the core-only CI job
    nb = None  # type: ignore[assignment]

    def _njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(func):
            return func
        return deco

ALLOWED_DISTS = Union[Phat, scipy.stats._distn_infrastructure.rv_frozen]


# Fitted models from `arch` and `pmdarima` are recognised by duck typing so that
# neither package has to be installed, or imported, to use Garchcaster.
def _is_arch_result(obj) -> bool:
    """A fitted ``arch`` model (``arch.univariate.base.ARCHModelResult``)."""
    needed = ('model', 'params', 'std_resid', 'conditional_volatility', 'resid')
    return all(hasattr(obj, name) for name in needed)


def _is_constant_mean(model) -> bool:
    """``arch.univariate.mean.ConstantMean``: a mean model without lags."""
    return type(model).__name__ == 'ConstantMean' or getattr(model, 'lags', None) is None


def _is_pmdarima_arima(obj) -> bool:
    """A fitted ``pmdarima.ARIMA``."""
    return hasattr(obj, 'order') and callable(getattr(obj, 'params', None)) and hasattr(obj, 'arima_res_')


###### Numba helper functions to improve GarchForecast speed ######
@_njit(cache=True)
def _step_sum(
    params:np.ndarray,
    values:np.ndarray
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

@_njit(cache=True)
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

@_njit(cache=True)
def _garch_one_step(
    vols:np.ndarray,
    resids:np.ndarray,
    innov:float,
    constant:float,
    arch_params:np.ndarray,
    garch_params:np.ndarray,
    p:int,
    q:int
    ) -> tuple[float, float]:
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
    params: np.ndarray

    @property
    def order(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def properties(self) -> tuple:
        raise NotImplementedError

    @property
    def constant(self) -> float:
        return float(self.params[0])

    @property
    def max_order(self) -> int:
        return max(self.order)

    @property
    def props(self) -> tuple:
        return self.properties

    @staticmethod
    def _as_params(params: Any) -> np.ndarray:
        """``None``, a sequence or a pandas Series of parameters as a float array."""
        if params is None:
            return np.zeros(0)
        if hasattr(params, 'to_numpy'):
            params = params.to_numpy()
        return np.asarray(params, dtype=float).ravel()

    def _infer_order(self, params) -> tuple[int, int]:
        """
        Infers the order of an ARMA or GARCH process.

        If params is even, assume even split between the params. If odd, assumes modulus lag belongs to
        first order.

        Parameters
        -----------
        params:     iterable; params of process from which order is inferred

        Return
        -------
        order:      tuple; 2x1 tuple with ARMA order (p,q) or GARCH order (m,n)
        """
        if params is None or getattr(params, 'empty', False) or not np.any(params):
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
        params:Any,
        order:tuple[int, int] | None=None,
        ):
        self.params = self._as_params(params)
        self._m, self._n = (
            self._infer_order(self.params[1:]) if order is None else (int(order[0]), int(order[1]))
        )

    @property
    def m(self) -> int:
        return self._m

    @property
    def n(self) -> int:
        return self._n

    @property
    def order(self) -> tuple[int, int]:
        return self.m, self.n

    @property
    def ar_params(self) -> np.ndarray:
        return self.params[1:1+self.m]

    @property
    def ma_params(self) -> np.ndarray:
        return self.params[1+self.m:1+self.m+self.n]

    @property
    def properties(self) -> tuple:
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
        params:Any,
        order:tuple[int, int] | None=None
        ):
        self.params = self._as_params(params)
        self._p, self._q = (
            self._infer_order(self.params[1:]) if order is None else (int(order[0]), int(order[1]))
        )

    @property
    def p(self) -> int:
        return self._p

    @property
    def q(self) -> int:
        return self._q

    @property
    def order(self) -> tuple[int, int]:
        return self.p, self.q

    @property
    def arch_params(self) -> np.ndarray:
        return self.params[1:1+self.q]

    @property
    def garch_params(self) -> np.ndarray:
        return self.params[1+self.q:1+self.q+self.p]

    @property
    def properties(self) -> tuple:
        return self.constant, self.arch_params, self.garch_params, self.p, self.q

class GarchcastResults:
    """
    Container for outputs from Garchcast.forecast

    `plot` method provides several options for plotting results

    Parameters
    -----------
    values:     np.ndarray; 2d n_paths x horizon array of values from Garchcast projection
    vols:       np.ndarray; 2d n_paths x horizon array of forecast volatilities
    resids:     np.ndarray; 2d n_paths x horizon array of forecast residuals relative to ARMA-GARCH
    """
    # 0.0.x attribute names
    iters = renamed_attribute('n_paths')
    periods = renamed_attribute('horizon')

    def __init__(self,
        values:np.ndarray,
        vols:np.ndarray,
        resids:np.ndarray
        ):
        self.values = values
        self.vols = vols
        self.resids = resids
        self._sim: PriceSim | None = None

        self.n_paths, self.horizon = values.shape

    @property
    def rets(self) -> np.ndarray:
        return 1 + self.values/100

    @property
    def vars(self) -> np.ndarray:
        return self.vols**2

    @property
    def sim(self) -> PriceSim:
        if self._sim is None:
            raise ValueError('You must plot a price simulation or call `_make_sim` directly')
        return self._sim

    @argsetter(['n_paths', 'horizon'])
    def _make_sim(self, n_paths:int, horizon:int, p:float=1) -> PriceSim:
        """
        Creates a price simulator used to generate images
        """
        self._sim = PriceSim(p0=p, periods=horizon, n=n_paths)
        return self._sim

    def plot(
        self,
        kind:str='vol',
        ax:Any=None,
        *args: Any,
        p: float = 1.0,
        n: int = 1,
        **kwargs: Any
        ) -> Any:
        """
        Main method used to generate plots. Needs matplotlib (the ``plot`` extra).

        Params
        -------
        kind:    str; kind of plot. Accepts 'vol', 'var', 'price', or 'end_price'
        ax:      matplotlib Axes object(s); if one is not provided, it will be created.
        p:       float; starting price for 'price' and 'end_price'
        n:       int; number of simulated paths to draw for 'price'
        *args, **kwargs:    forwarded to the matplotlib call ('price' and 'end_price' take keywords only)

        Return
        -------
        ax:      matplotlib Axes object(s); for 'end_price' a tuple ``(ax, prices, bins)``
        """
        import matplotlib.pyplot as plt
        multi_ax = ['price']
        if ax is None and kind not in multi_ax:
            fig, ax = plt.subplots(1,1,figsize=(10,6))

        if kind == 'vol':
            ax = self._plot_vol(ax, *args, **kwargs)
        elif kind == 'var':
            ax = self._plot_var(ax, *args, **kwargs)
        elif kind == 'price':
            ax = self._plot_price(p, n, ax, **kwargs)
        elif kind == 'end_price':
            return self._plot_end_price(p, ax, **kwargs)
        else:
            raise ValueError(f'`kind:` {kind} not supported')

        return ax

    def _plot_vol(self, ax:Any, *args, **kwargs) -> Any:
        import matplotlib.pyplot as plt
        ax.plot((self.vols).mean(axis=0))
        ax.set_xlabel('Days')
        plt.suptitle('Phat-GARCH Forecast: Conditional Volatility')

        return ax

    def _plot_var(self, ax:Any, *args, **kwargs) -> Any:
        import matplotlib.pyplot as plt
        ax.plot((self.vols**2).mean(axis=0), *args, **kwargs)
        ax.set_xlabel('Days')
        plt.suptitle('Phat-GARCH Forecast: Conditional Variance')

        return ax

    def _plot_price(self,
        p:float,
        n:int,
        axes:Any,
        **kwargs
        ) -> Any:
        sim = self._make_sim(p=p)
        rng = _rng(kwargs.pop('random_state', None))

        all_axes = []
        for i in rng.choice(self.n_paths, size=min(int(n), self.n_paths), replace=False):
            # keyword call: argsetter maps positional arguments onto (p0, rets, horizon) only
            _, __, chart_axes = sim.sim(
                rets=1 + self.values[i]/100, periods=self.horizon, show_chart=True, axes=axes, **kwargs
            )
            all_axes.append(chart_axes)

        if axes is None:
            return np.array(all_axes)
        else:
            return axes

    def _plot_end_price(self,
        p:float,
        ax:Any,
        **kwargs
        ) -> tuple[Any, np.ndarray, np.ndarray]:
        sim = self._make_sim(p=p)
        _, P, (ax, bins) = sim.sims(rets=1 + self.values/100, show_chart=True, ax=ax, **kwargs)
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
        history of standardized residuals after filtering an ARMA-GARCH. Only required if
        garch is not ARCHModelResult type.
    vols: float or iterable of floats
        history of conditional volatilities resulting from an ARMA-GARCH process. Only required if
        garch is not ARCHModelResult type.
    resids: float or iterable of floats
        history of residuals after filtering an ARMA-GARCH. Only required if garch is not an
        ARCHModelResult.
    n_paths: int
        number of simulated paths
    horizon: int
        number of periods forecast along each path
    dist: Phat or scipy rv_frozen class
        distribution used to generate residuals
    use_backcast: bool
        use procedure to smooth values. See `here <https://arch.readthedocs.io/en/latest/univariate/generated/generated/arch.univariate.GARCH.backcast.html>`_.
    """
    GARCH_HAS_MEAN_PROCESS = False

    # 0.0.x attribute names
    iters = renamed_attribute('n_paths')
    periods = renamed_attribute('horizon')

    @renamed_kwargs(iters='n_paths', periods='horizon')
    def __init__(self,
        garch:Union[ARCHModelResult, Iterable[float], None]=None,
        arma:Union[pmdarima.ARIMA, Iterable[float], None]=None,
        y:Iterable[float] | None=None,
        vols:Iterable[float] | None=None,
        resids:Iterable[float] | None=None,
        n_paths:int=1,
        horizon:int=252,
        order:tuple[int, ...] | None=None,
        dist:ALLOWED_DISTS | None=None,
        use_backcast:bool=True,
        ):
        garch_any: Any = garch   # fitted models are duck-typed; see _is_arch_result
        arma_any: Any = arma
        mean_order: tuple[int, int] | None
        vol_order: tuple[int, int] | None

        ### First, set self.mean
        if _is_pmdarima_arima(arma_any):
            m, _, n = arma_any.order
            mean_order = (m,n)
            mean_params = arma_any.params()[:5]
        elif _is_arch_result(garch_any):
            if _is_constant_mean(garch_any.model):
                mean_order = (0, 0)
                mean_params = garch_any.params[:1]
            else:
                mean_order = (garch_any.model.lags[0].size, 0)
                mean_params = garch_any.params[:mean_order[0]+1]
                self.GARCH_HAS_MEAN_PROCESS = True
        elif arma is not None:
            mean_order = (int(order[0]), int(order[1])) if order is not None else None
            mean_params = arrayarize(arma_any)
        else:
            raise ValueError(
                'Provide `arma` (parameters or a fitted model) or a fitted `arch` model as `garch`'
            )

        self.mean = MEAN(mean_params, order=mean_order)

        ### Second, set self.vol
        if _is_arch_result(garch_any):
            if hasattr(garch_any.model.volatility, 'p'):
                vol_order = (garch_any.model.volatility.p, garch_any.model.volatility.q)
            else:
                vol_order = (0, 0)

            first = self.mean.m + 1 if self.GARCH_HAS_MEAN_PROCESS else 1
            vol_params = garch_any.params[first:]
        elif garch is not None:
            vol_params = arrayarize(garch_any)
            vol_order = (int(order[2]), int(order[3])) if order is not None else None
        else:
            raise ValueError('Provide `garch` (parameters or a fitted arch model)')

        self.vol = VOL(vol_params, order=vol_order)
        self.order = (*self.mean.order, *self.vol.order)

        if order is not None and order != self.order:
            msg = 'The `order` parameter provided does not match the order'
            msg += ' inferred from the models provided and has been OVERRIDDEN. The `order` parameter'
            msg += ' is not required when passing `arch` and/or pmdarima models to Garchcaster.'
            warnings.warn(msg, stacklevel=2)

        self.maxlag = max(self.order)

        ### Third, assign values
        if _is_arch_result(garch_any):
            if any([y is not None, vols is not None, resids is not None]):
                raise ValueError('If `garch` is ARCHModelResult do not provide `y`, `vol`, or `resids`')
            y_arr = np.asarray(garch_any.std_resid, dtype=float)[self.maxlag:].copy()
            vols_arr = np.asarray(garch_any.conditional_volatility, dtype=float)[self.maxlag:].copy()
            resids_arr = np.asarray(garch_any.resid, dtype=float)[self.maxlag:].copy()
        else:
            if y is None or vols is None or resids is None:
                raise ValueError('Provide `y`, `vols` and `resids` when `garch` is not a fitted arch model')
            y_arr = arrayarize(y).astype(float)
            vols_arr = arrayarize(vols).astype(float)
            resids_arr = arrayarize(resids).astype(float)

        self.horizon = horizon
        self.n_paths = n_paths

        self.y_hist = y_arr[-self.maxlag:]
        self.vols_hist = vols_arr[-self.maxlag:]
        self.resids_hist = (
            self._backcast(resids_arr, self.maxlag) if use_backcast else resids_arr[-self.maxlag:]
        )

        self._dist = dist

    @property
    def dist(self) -> ALLOWED_DISTS | None:
        """Innovation distribution; ``None`` means standard Gaussian innovations."""
        return self._dist

    @argsetter(['n_paths', 'horizon'], flat=True)
    def _sample_innovs(self,
        n_paths:int | None=None,
        horizon:int | None=None,
        dist:ALLOWED_DISTS | None=None,
        seed:Any=None,
        random_state:Any=None,
        ) -> np.ndarray:
        """
        Generates innovations according to the provided distribution. If a distribution is not
        provided, default is Gaussian.

        A :class:`~phat.Phat` innovation distribution is sampled with its exact
        sampler, :meth:`~phat.Phat.std_rvs`, so the innovations have unit
        variance (this needs both tail shapes below 1/2).

        Params
        -------
        n_paths:        int; number of simulated paths
        horizon:        int; number of periods forecast along each path
        dist:           Phat or scipy rv_frozen class; distribution used to generate residuals
        random_state:   ``None``, int seed, ``numpy.random.Generator`` or ``RandomState``
        seed:           alias of ``random_state``

        Return
        -------
        innovs:         np.ndarray
        """
        shape = (int(n_paths), int(horizon))  # type: ignore[arg-type]  # filled by argsetter
        rng = _rng(random_state if random_state is not None else seed)
        dist = dist if dist is not None else self._dist
        if dist is None:
            return rng.standard_normal(shape)
        rvfunc = dist.std_rvs if hasattr(dist, 'std_rvs') else dist.rvs
        return np.asarray(rvfunc(size=shape, random_state=rng), dtype=float)

    def _backcast(self, resids: np.ndarray, maxlag: int) -> np.ndarray:
        power = 2
        tau = min(75, resids.shape[0])
        w = 0.94 ** np.arange(tau)
        w = w / sum(w)
        backcast = np.sum((abs(resids[:tau]) ** power) * w)

        return np.repeat(backcast, maxlag)

    @staticmethod
    @_njit(cache=True)
    def _forecast_looper(
        y:np.ndarray,
        vols:np.ndarray,
        resids:np.ndarray,
        innovs:np.ndarray,
        n_paths:int,
        horizon:int,
        mean_args:tuple,
        vol_args:tuple,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        y:              np.ndarray; historical standardized residuals of ARMA-GARCH fit, last `maxlag` periods
        vols:           np.ndarray; historical volatilities of ARMA-GARCH fit up to `maxlag` periods ago
        resids:         np.ndarray; historical residuals of ARMA-GARCH fit up to `maxlag` periods ago
        innovs:         n_paths x horizon np.ndarray of garch residuals
        n_paths:        int; number of simulated paths
        horizon:        int; number of periods forecast along each path
        dist:           Phat or scipy rv_frozen class; distribution used to generate residuals
        mean_args:      tuple; properties of the mean process use in `_arma_one_step`
        vol_args:       tuple; properties of the vol process use in `_garch_one_step`
        """
        y_ = np.zeros((n_paths, horizon))
        vols_ = np.zeros((n_paths, horizon))
        resids_ = np.zeros((n_paths, horizon))
        for i in range(n_paths):
            y_i = y.copy()
            vols_i = vols.copy()
            resids_i = resids.copy()
            for t in range(horizon):
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

    @renamed_kwargs(iters='n_paths', periods='horizon')
    @argsetter(['dist', 'n_paths', 'horizon'], flat=True)
    def forecast(self,
        n_paths:int | None=None,
        horizon:int | None=None,
        dist:ALLOWED_DISTS | None=None,
        innovs:Iterable[float] | None=None,
        seed:Any=None,
        random_state:Any=None,
        ) -> GarchcastResults:
        """
        User-facing interface with forecast generator

        Params:
            n_paths: int
                number of simulated paths
            horizon: int
                number of periods forecast along each path
            dist: Phat or scipy rv_frozen class
                distribution used to generate residuals
            innovs: np.ndarray
                n_paths x horizon of garch residuals
            random_state: None, int seed, numpy.random.Generator or RandomState
                for the innovations if `innovs` is not provided
            seed: alias of `random_state`
        """
        n_iters, n_periods = int(n_paths), int(horizon)  # type: ignore[arg-type]  # filled by argsetter
        if innovs is None:
            innovs_arr = self._sample_innovs(
                n_iters, n_periods, dist=dist, seed=seed, random_state=random_state
            )
        else:
            innovs_arr = np.asarray(innovs, dtype=float)
            if innovs_arr.shape != (n_iters, n_periods):
                raise ValueError(f'`innovs` must have shape {(n_iters, n_periods)}, got {innovs_arr.shape}')

        results = Garchcaster._forecast_looper(
            self.y_hist,
            self.vols_hist,
            self.resids_hist,
            innovs_arr,
            n_iters, n_periods,
            self.mean.props,
            self.vol.props
        )
        return GarchcastResults(*results)
