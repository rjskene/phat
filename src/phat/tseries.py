"""
Custom functionality for generating ARMA-GARCH time-series forecasts.
"""
from __future__ import annotations

import re
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


def _arch_mean(result) -> tuple[np.ndarray, int, int]:
    """
    ``(c, phi_1..phi_m)``, ``m`` and the number of mean parameters of a fitted
    ``arch`` model.

    Zero-mean, constant-mean and AR models with consecutive lags are supported.
    The mean is read from the model's parameter names: ``Const`` or ``mu`` for
    the constant and ``name[k]`` for lag ``k``.
    """
    model = result.model
    if getattr(model, 'x', None) is not None:
        raise NotImplementedError('Garchcaster does not support mean models with exogenous regressors')
    names = list(model.parameter_names())
    has_constant = bool(names) and names[0] in ('Const', 'mu')
    lag_names = names[1:] if has_constant else names
    lags = [re.fullmatch(r'.*\[(\d+)\]', name) for name in lag_names]
    m = len(lags)
    found = [int(lag.group(1)) for lag in lags if lag is not None]
    if len(found) != m or found != list(range(1, m + 1)):
        raise NotImplementedError('Garchcaster supports AR mean models with consecutive lags 1..m only')
    params = np.asarray(result.params, dtype=float)[:len(names)]
    return (params.copy() if has_constant else np.concatenate(([0.0], params))), m, len(names)


def _arch_vol_order(volatility) -> tuple[int, int]:
    """
    ``(p, q)`` of a fitted ``arch`` volatility process in this module's convention:
    ``p`` lagged variances and ``q`` lagged squared residuals (``arch`` calls
    them ``q`` and ``p``).
    """
    name = type(volatility).__name__
    if name == 'ConstantVariance':
        return (0, 0)
    if name not in ('GARCH', 'ARCH'):
        raise NotImplementedError(f'Garchcaster does not support the {name} volatility process')
    if getattr(volatility, 'o', 0) != 0 or getattr(volatility, 'power', 2.0) != 2.0:
        raise NotImplementedError(
            'Garchcaster simulates symmetric GARCH on variances only; '
            'asymmetric terms (o > 0) and powers other than 2 are not supported'
        )
    return (int(getattr(volatility, 'q', 0)), int(volatility.p))


def _is_pmdarima_arima(obj) -> bool:
    """A fitted ``pmdarima.ARIMA``."""
    return hasattr(obj, 'order') and callable(getattr(obj, 'params', None)) and hasattr(obj, 'arima_res_')


def _check_same_end(arima, result) -> None:
    """
    The series comes from the ARMA model, the residuals and volatilities from
    the GARCH model; a forecast needs all three to end on the same observation.
    """
    resid = getattr(arima, 'resid', None)
    if not callable(resid):
        return
    arma_resid = np.asarray(resid(), dtype=float).ravel()
    garch_input = np.asarray(result.model.y, dtype=float).ravel()
    k = min(5, arma_resid.size, garch_input.size)
    if k and not np.allclose(arma_resid[-k:], garch_input[-k:], rtol=1e-8, atol=1e-10):
        raise ValueError(
            '`garch` must be fitted to the residuals of `arma` up to its last observation: '
            'the two models do not end on the same observation'
        )


def _pmdarima_mean(arima) -> tuple[np.ndarray, tuple[int, int]]:
    """``(c, phi_1..phi_m, theta_1..theta_n)`` and ``(m, n)`` of a fitted ``pmdarima.ARIMA``."""
    m, d, n = (int(v) for v in arima.order)
    if d != 0:
        raise NotImplementedError('Garchcaster does not support differenced ARIMA models (d > 0)')
    seasonal = getattr(arima, 'seasonal_order', None)
    if seasonal is not None and any(seasonal[:3]):
        raise NotImplementedError('Garchcaster does not support seasonal ARIMA models')
    params = np.asarray(arima.params(), dtype=float)
    names = getattr(arima.arima_res_, 'param_names', None)
    has_constant = 'intercept' in names if names else bool(getattr(arima, 'with_intercept', True))
    if has_constant:
        return params[:1 + m + n].copy(), (m, n)
    return np.concatenate(([0.0], params[:m + n])), (m, n)


@_njit(cache=True)
def _lag_sum(
    params:np.ndarray,
    values:np.ndarray
    ) -> float:
    """
    Summation term of an AR(m), MA(n), ARCH(q) or GARCH(p) process:

    .. math::

        \\sum_{k=1}^{K} \\text{params}_k \\, \\text{values}_{t-k}

    ``values`` is a history running from oldest to newest, so lag ``k`` is its
    ``k``-th value from the end and ``params[k-1]`` is the coefficient of lag ``k``.

    Params
    -------
    params:     np.ndarray; fitted coefficients, lag 1 first
    values:     np.ndarray; history of the lagged quantity, most recent value last
    """
    total = 0.0
    last = values.shape[0] - 1
    for k in range(params.shape[0]):
        total += params[k]*values[last - k]

    return total

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
    y:              np.ndarray; last `maxlag` values of the series, most recent last
    resids:         np.ndarray; last `maxlag` residuals of the ARMA-GARCH fit, most recent last
    constant:       float; constant of the mean process
    ar_params:      np.ndarray; paramters of the AR process
    ma_params:      np.ndarray; paramters of the MA process
    m:              int; order of the AR process
    n:              int; order of the MA process
    """
    ar = _lag_sum(ar_params, y) if m != 0 else 0.0
    ma = _lag_sum(ma_params, resids) if n != 0 else 0.0

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
    arch = _lag_sum(arch_params, resids**2) if q != 0 else 0.0
    garch = _lag_sum(garch_params, vols**2) if p != 0 else 0.0

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
    def gross_returns(self) -> np.ndarray:
        """Price relatives ``1 + values/100``; ``values`` are returns in percent."""
        return 1 + self.values/100

    rets = renamed_attribute('gross_returns')

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
        Plot the forecast. Needs matplotlib (the ``plot`` extra).

        Parameters
        ----------
        kind : str
            ``'vol'``, ``'var'``, ``'price'`` or ``'end_price'``
        ax : matplotlib Axes, optional
            created when not given
        p : float
            starting price for ``'price'`` and ``'end_price'``
        n : int
            number of simulated paths to draw for ``'price'``
        ``*args``, ``**kwargs``
            forwarded to the matplotlib call (``'price'`` and ``'end_price'``
            take keywords only)

        Returns
        -------
        matplotlib Axes; for ``'end_price'`` a tuple ``(ax, prices, bins)``
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
Simulates an ARMA(m, n)-GARCH(p, q) process forward from the end of a sample:

.. math::

    y_t = c + \\sum_{i=1}^m \\phi_i y_{t-i} + \\sum_{j=1}^n \\theta_j \\epsilon_{t-j} + \\epsilon_t,
    \\qquad \\epsilon_t = \\sigma_t z_t,

    \\sigma_t^2 = \\omega + \\sum_{k=1}^q \\alpha_k \\epsilon_{t-k}^2 + \\sum_{l=1}^p \\beta_l \\sigma_{t-l}^2

with unit-variance innovations :math:`z_t`. Every path starts from the last
observed values, residuals and conditional volatilities of the sample, so on
the same innovations a forecast built from a fitted ``arch`` model equals that
model's own simulation forecast.

``p`` counts the lagged variances and ``q`` the lagged squared residuals, as in
Bollerslev (1986). The ``arch`` package uses the two letters the other way
round; models fitted with ``arch`` are translated.

Fitted models can be passed directly: a ``pmdarima`` ``ARIMA`` as `arma` and/or
an ``arch`` result as `garch`. Otherwise give the parameters and the histories.
Supported fitted models are zero-mean, constant-mean and AR mean models with
consecutive lags and no exogenous regressors, constant variance, ARCH and
symmetric GARCH, and ARMA models without differencing or seasonal terms.

Params:
    garch: fitted ``arch`` model (``ARCHModelResult``) OR iterable of floats
        The parameters are ordered ``omega, alpha_1..alpha_q, beta_1..beta_p``.
        When `arma` is also given, the ``arch`` model is the one fitted to the
        ARMA residuals; only its volatility parameters are used, its mean is not.
    arma: fitted ``pmdarima.ARIMA`` OR iterable of floats
        The parameters are ordered ``c, phi_1..phi_m, theta_1..theta_n``. If
        ``None`` the mean process is taken from `garch`.
    y:  iterable of floats
        The series being forecast, oldest value first; the last ``m`` values
        start the AR recursion. Required when `garch` is given as parameters,
        and when `arma` is given as parameters with a fitted `garch`.
    vols: iterable of floats
        Conditional volatilities of the fit, oldest first. Only with `garch`
        given as parameters.
    resids: iterable of floats
        Residuals :math:`\\epsilon_t` of the fit (not standardized), oldest first.
        Only with `garch` given as parameters.
    n_paths: int
        number of simulated paths
    horizon: int
        number of periods forecast along each path
    order: tuple ``(m, n, p, q)``
        Only needed with parameters; inferred from fitted models.
    dist: Phat or scipy rv_frozen class
        distribution of the innovations; standard Gaussian when ``None``
    use_backcast: bool
        Deprecated and without effect. In 0.0.x the default replaced the last
        residuals with a weighted average of the squared residuals at the
        *start* of the sample, which made the first forecast variance wrong.
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
        use_backcast:bool | None=None,
        ):
        if use_backcast is not None:
            warnings.warn(
                '`use_backcast` is deprecated and has no effect: a forecast starts from the last '
                'residuals and conditional volatilities of the sample.',
                DeprecationWarning, stacklevel=3,
            )
        garch_any: Any = garch   # fitted models are duck-typed; see _is_arch_result
        arma_any: Any = arma
        mean_order: tuple[int, int] | None
        vol_order: tuple[int, int] | None
        garch_is_fitted = _is_arch_result(garch_any)
        arma_is_fitted = _is_pmdarima_arima(arma_any)
        n_arch_mean = 0   # leading parameters of a fitted arch model that belong to its mean

        ### First, set self.mean
        if garch_is_fitted:
            arch_mean_params, arch_lags, n_arch_mean = _arch_mean(garch_any)

        if arma_is_fitted:
            mean_params, mean_order = _pmdarima_mean(arma_any)
        elif arma is not None:
            mean_order = (int(order[0]), int(order[1])) if order is not None else None
            mean_params = arrayarize(arma_any)
        elif garch_is_fitted:
            mean_order = (arch_lags, 0)
            mean_params = arch_mean_params
            self.GARCH_HAS_MEAN_PROCESS = arch_lags > 0
        else:
            raise ValueError(
                'Provide `arma` (parameters or a fitted model) or a fitted `arch` model as `garch`'
            )

        self.mean = MEAN(mean_params, order=mean_order)

        ### Second, set self.vol
        if garch_is_fitted:
            vol_order = _arch_vol_order(garch_any.model.volatility)
            vol_params = np.asarray(garch_any.params, dtype=float)[n_arch_mean:]
        elif garch is not None:
            vol_params = arrayarize(garch_any)
            vol_order = (int(order[2]), int(order[3])) if order is not None else None
        else:
            raise ValueError('Provide `garch` (parameters or a fitted arch model)')

        self.vol = VOL(vol_params, order=vol_order)
        self.order = (*self.mean.order, *self.vol.order)

        if order is not None and tuple(order) != self.order:
            msg = 'The `order` parameter provided does not match the order'
            msg += ' inferred from the models provided and has been OVERRIDDEN. The `order` parameter'
            msg += ' is not required when passing `arch` and/or pmdarima models to Garchcaster.'
            warnings.warn(msg, stacklevel=3)

        self.maxlag = max(self.order)

        ### Third, the histories each path starts from
        if garch_is_fitted:
            if vols is not None or resids is not None:
                raise ValueError('If `garch` is a fitted arch model do not provide `vols` or `resids`')
            vols_arr = np.asarray(garch_any.conditional_volatility, dtype=float)
            resids_arr = np.asarray(garch_any.resid, dtype=float)
            if arma_is_fitted:
                if y is not None:
                    raise ValueError('If `arma` is a fitted model do not provide `y`')
                y_arr = np.asarray(arma_any.arima_res_.model.endog, dtype=float).ravel()
                _check_same_end(arma_any, garch_any)
            elif arma is not None:
                if y is None and self.mean.m > 0:
                    raise ValueError('Provide `y`, the series the ARMA parameters were fitted to')
                y_arr = arrayarize(y).astype(float) if y is not None else resids_arr
            else:
                if y is not None:
                    raise ValueError('If `garch` is a fitted arch model with its own mean do not provide `y`')
                y_arr = np.asarray(garch_any.model.y, dtype=float)
        else:
            if y is None or vols is None or resids is None:
                raise ValueError('Provide `y`, `vols` and `resids` when `garch` is not a fitted arch model')
            y_arr = arrayarize(y).astype(float)
            vols_arr = arrayarize(vols).astype(float)
            resids_arr = arrayarize(resids).astype(float)

        self.horizon = horizon
        self.n_paths = n_paths

        keep = max(self.maxlag, 1)
        for name, arr, need in (
            ('y', y_arr, self.mean.m),
            ('vols', vols_arr, self.vol.p),
            ('resids', resids_arr, max(self.mean.n, self.vol.q)),
        ):
            tail = arr[-need:] if need else arr[:0]
            if arr.size < need or not np.isfinite(tail).all():
                raise ValueError(
                    f'`{name}` needs at least {need} finite trailing values for order {self.order}'
                )
        self.y_hist = self._tail(y_arr, keep)
        self.vols_hist = self._tail(vols_arr, keep)
        self.resids_hist = self._tail(resids_arr, keep)

        self._dist = dist

    @staticmethod
    def _tail(arr: np.ndarray, keep: int) -> np.ndarray:
        """The last `keep` values, padded with zeros in front when the history is shorter."""
        out = np.zeros(keep)
        tail = arr[-keep:]
        out[keep - tail.size:] = tail
        return np.where(np.isfinite(out), out, 0.0)

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
        y:              np.ndarray; last `maxlag` values of the series, most recent last
        vols:           np.ndarray; last `maxlag` conditional volatilities, most recent last
        resids:         np.ndarray; last `maxlag` residuals, most recent last
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

                # drop the oldest value of each history, then append the new one
                for k in range(y_i.size - 1):
                    y_i[k] = y_i[k + 1]
                    vols_i[k] = vols_i[k + 1]
                    resids_i[k] = resids_i[k + 1]
                y_i[-1] = y_[i, t]
                vols_i[-1] = vols_[i, t]
                resids_i[-1] = resids_[i, t]
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
        Simulate ``n_paths`` paths of ``horizon`` periods from the ARMA-GARCH process.

        Parameters
        ----------
        n_paths : int, optional
            number of simulated paths (default: the value given to the constructor)
        horizon : int, optional
            number of periods forecast along each path (default: as constructed)
        dist : Phat or frozen scipy.stats distribution, optional
            innovation distribution; standard Gaussian when ``None``. A
            :class:`~phat.dists.Phat` is sampled with its exact unit-variance
            sampler :meth:`~phat.dists.Phat.std_rvs`.
        innovs : array of shape ``(n_paths, horizon)``, optional
            innovations to use instead of drawing them
        random_state : None, int, numpy.random.Generator or RandomState
            for the innovations when ``innovs`` is not given
        seed
            alias of ``random_state``

        Returns
        -------
        GarchcastResults
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
