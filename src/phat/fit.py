"""
Maximum-likelihood estimation of the Phat distribution.

:func:`fit_phat` (reached through :meth:`phat.Phat.fit`) maximises the
likelihood with ``scipy.optimize`` and reports standard errors from the inverse
observed information (a finite-difference Hessian at the optimum). It needs
NumPy and SciPy only.

:class:`PhatFit` is the older ``statsmodels`` ``GenericLikelihoodModel``
wrapper, kept for callers who want that results API; it is used when
``Phat.fit(..., backend='statsmodels')`` is requested and needs the ``fit``
extra (``pip install "phat-tails[fit]"``). ``statsmodels`` is imported only on
that path.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.optimize as opt
import scipy.stats as scist

from phat.dists import Phat

__all__ = ['PhatFitResult', 'fit_phat', 'PhatFit']  # noqa: F822  # PhatFit via __getattr__

PARAM_NAMES = ('mu', 'sigma', 'xi_left', 'xi_right')
_XI_MAX = 10.0
_BOUND_TOL = 1e-8


@dataclass
class PhatFitResult:
    """
    Result of a maximum-likelihood fit of :class:`~phat.Phat`.

    Attributes
    ----------
    params:     estimates of the free parameters, in :attr:`names` order
    names:      names of the free parameters (``mu``, ``sigma`` and, unless fixed, ``xi_left``, ``xi_right``)
    bse:        standard errors (square roots of the diagonal of :attr:`cov`); ``nan``
                for an estimate that sits on a bound (see :attr:`at_bound`) and where
                the observed information is not invertible
    cov:        inverse observed-information matrix of the estimates that are not on a
                bound; ``nan`` in the rows and columns of those that are
    llf:        log-likelihood at the optimum
    nobs:       number of observations
    fixed:      parameters held fixed during the fit (``xi_left``, ``xi_right`` and ``weights``)
    converged:  whether the optimiser reported success
    message:    the optimiser's message
    at_bound:   names of the estimates that sit on a bound of the search, typically a
                tail shape at ``xi = 0``. The usual standard errors and Wald intervals
                do not apply to them, so none are reported.
    """

    params: np.ndarray
    names: tuple[str, ...]
    bse: np.ndarray
    cov: np.ndarray
    llf: float
    nobs: int
    fixed: dict[str, Any] = field(default_factory=dict)
    converged: bool = True
    message: str = ''
    at_bound: tuple[str, ...] = ()

    @property
    def params_dict(self) -> dict[str, float]:
        """``{name: estimate}`` for the free parameters."""
        return {name: float(v) for name, v in zip(self.names, self.params)}

    @property
    def all_params(self) -> dict[str, float]:
        """``mu, sigma, xi_left, xi_right`` with fixed values filled in."""
        out = dict(self.params_dict)
        for name in ('xi_left', 'xi_right'):
            if name in self.fixed:
                out[name] = float(self.fixed[name])
        return {name: out[name] for name in PARAM_NAMES}

    @property
    def dist(self) -> Phat:
        """The fitted distribution."""
        return Phat(**self.all_params, weights=self.fixed.get('weights'))

    @property
    def aic(self) -> float:
        return 2 * len(self.params) - 2 * self.llf

    @property
    def bic(self) -> float:
        return len(self.params) * np.log(self.nobs) - 2 * self.llf

    def conf_int(self, alpha: float = 0.05) -> np.ndarray:
        """Wald confidence intervals, one row per free parameter."""
        z = scist.norm.ppf(1 - alpha / 2)
        return np.column_stack([self.params - z * self.bse, self.params + z * self.bse])

    def summary(self) -> str:
        lines = [f'Phat maximum-likelihood fit  (nobs = {self.nobs}, llf = {self.llf:.4f})']
        lines.append(f'{"":8} {"estimate":>12} {"std err":>12}')
        for name, est, se in zip(self.names, self.params, self.bse):
            err = '(at bound)' if name in self.at_bound else f'{se:.6f}'
            lines.append(f'{name:8} {est:12.6f} {err:>12}')
        for name, value in self.fixed.items():
            shown = f'{value:.6f}' if isinstance(value, float) else str(value)
            lines.append(f'{name:8} {shown:>12} {"(fixed)":>12}')
        if not self.converged:
            lines.append(f'WARNING: optimiser did not converge: {self.message}')
        return '\n'.join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _start_values(x: np.ndarray, free_tails: bool) -> np.ndarray:
    """
    Robust starting point: the median for ``mu``; the interquartile range for
    ``sigma`` (for ``xi`` around 0.2 the Phat IQR is about ``2.6 sigma``); ``0.2``
    for a free tail shape.
    """
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    sig0 = max((q3 - q1) / 2.6, 1e-6 * (abs(med) + 1))
    start = [med, sig0]
    if free_tails:
        start += [0.2, 0.2]
    return np.array(start, dtype=float)


def _numerical_hessian(f: Callable[[np.ndarray], float], theta: np.ndarray) -> np.ndarray:
    """Central-difference Hessian of a scalar function. Entries are ``nan`` where ``f`` is not finite."""
    k = theta.size
    h = 1e-4 * np.maximum(1.0, np.abs(theta))
    H = np.empty((k, k))
    f0 = f(theta)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        H[i, i] = (f(theta + ei) - 2 * f0 + f(theta - ei)) / h[i] ** 2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (
                f(theta + ei + ej) - f(theta + ei - ej) - f(theta - ei + ej) + f(theta - ei - ej)
            ) / (4 * h[i] * h[j])
    return H


def fit_phat(
    values: Iterable[float],
    xi_left: float | None = None,
    xi_right: float | None = None,
    weights: Sequence[float] | None = None,
    start: Sequence[float] | None = None,
    **minimize_kwargs: Any,
) -> PhatFitResult:
    """
    Maximum-likelihood fit of :class:`~phat.Phat` with ``scipy.optimize.minimize``.

    Parameters
    ----------
    values:     observations
    xi_left, xi_right: tail shapes to hold fixed. Give both for the two-stage estimator
                (shapes from the Hill double bootstrap, then ``mu`` and ``sigma`` by
                likelihood); give neither for the full four-parameter fit.
    weights:    mixture weights to hold fixed (default ``(0.5, 0.5)``)
    start:      starting values for the free parameters; see :func:`_start_values`
    **minimize_kwargs:  forwarded to ``scipy.optimize.minimize`` (default method
                ``L-BFGS-B`` with ``sigma > 0`` and ``0 <= xi <= 10``)
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size < 3:
        raise ValueError('need at least three observations')
    if (xi_left is None) != (xi_right is None):
        raise ValueError('fix both tail shapes or neither')
    free_tails = xi_left is None

    def unpack(theta: np.ndarray) -> Phat:
        if free_tails:
            return Phat(theta[0], theta[1], theta[2], theta[3], weights=weights)
        assert xi_left is not None and xi_right is not None
        return Phat(theta[0], theta[1], float(xi_left), float(xi_right), weights=weights)

    def nll(theta: np.ndarray) -> float:
        try:
            value = float(np.sum(unpack(theta).nll(x)))
        except ValueError:
            return np.inf
        return value if np.isfinite(value) else np.inf

    theta0 = _start_values(x, free_tails) if start is None else np.asarray(start, dtype=float)
    bounds = [(None, None), (1e-12, None)] + ([(0.0, _XI_MAX)] * 2 if free_tails else [])
    kwargs: dict[str, Any] = {'method': 'L-BFGS-B', 'bounds': bounds}
    kwargs.update(minimize_kwargs)
    res = opt.minimize(nll, theta0, **kwargs)

    theta = np.asarray(res.x, dtype=float)
    names = PARAM_NAMES if free_tails else PARAM_NAMES[:2]

    # An estimate on a bound has no Wald standard error. The observed information
    # is taken over the remaining parameters, with the bound ones held there.
    on_bound = np.array([
        (lo is not None and value <= lo + _BOUND_TOL) or (hi is not None and value >= hi - _BOUND_TOL)
        for value, (lo, hi) in zip(theta, kwargs.get('bounds') or [(None, None)] * theta.size)
    ])
    inside = np.flatnonzero(~on_bound)

    def nll_inside(sub: np.ndarray) -> float:
        full = theta.copy()
        full[inside] = sub
        return nll(full)

    cov = np.full((theta.size, theta.size), np.nan)
    if inside.size:
        H = _numerical_hessian(nll_inside, theta[inside])
        if np.all(np.isfinite(H)):
            try:
                cov[np.ix_(inside, inside)] = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                pass
    with np.errstate(invalid='ignore'):
        bse = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))

    fixed: dict[str, Any] = {}
    if not free_tails:
        fixed['xi_left'], fixed['xi_right'] = float(xi_left), float(xi_right)  # type: ignore[arg-type]
    if weights is not None:
        fixed['weights'] = tuple(float(v) for v in weights)
    return PhatFitResult(
        params=theta, names=names, bse=bse, cov=cov, llf=-float(res.fun), nobs=int(x.size),
        fixed=fixed, converged=bool(res.success), message=str(res.message),
        at_bound=tuple(name for name, hit in zip(names, on_bound) if hit),
    )


def __getattr__(name: str) -> Any:
    if name == 'PhatFit':
        return _statsmodels_phatfit()
    raise AttributeError(f"module 'phat.fit' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {'PhatFit'})


_PHATFIT_CACHE: list = []


def _statsmodels_phatfit() -> type:
    """Build (once) the ``statsmodels`` ``GenericLikelihoodModel`` subclass."""
    if _PHATFIT_CACHE:
        return _PHATFIT_CACHE[0]
    try:
        from statsmodels.base.model import GenericLikelihoodModel
    except ImportError as exc:
        raise ImportError('PhatFit requires statsmodels: pip install "phat-tails[fit]"') from exc

    class PhatFit(GenericLikelihoodModel):  # type: ignore[misc]
        """
        ``statsmodels`` ``GenericLikelihoodModel`` for the Phat distribution.

        Parameters
        -----------
        endog:      iterable of observations
        exog:       a constant column (the model has no covariates)
        xi_left:    shape of the left tail. ``None`` to estimate it.
        xi_right:   shape of the right tail. ``None`` to estimate it.
        tail_est:   callable returning ``(xi_left, xi_right)`` from ``endog``; used
                    when both shapes are ``None``.
        """

        def __init__(self,
            endog: Iterable,
            exog: Iterable,
            xi_left: float | None = None,
            xi_right: float | None = None,
            tail_est: Callable | None = None,
            **kwds: Any):
            if xi_right is None and xi_left is None and tail_est is not None:
                self.xi_left, self.xi_right = tail_est(endog)
            else:
                self.xi_left, self.xi_right = xi_left, xi_right

            super().__init__(endog, exog, **kwds)

        def nloglikeobs(self, params):
            """Negative log-likelihood per observation. Infeasible parameters score ``inf``."""
            try:
                if len(params) == 4:
                    phatdist = Phat(*params)
                else:
                    phatdist = Phat(params[0], params[1], self.xi_left, self.xi_right)
            except ValueError:
                return np.full(np.shape(self.endog), np.inf)
            return phatdist.nll(self.endog)

        def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
            """Start values: see :func:`phat.fit._start_values`."""
            free_tails = self.xi_left is None and self.xi_right is None
            if start_params is None:
                start_params = _start_values(np.asarray(self.endog, dtype=float), free_tails)

            names = ['mu', 'sigma'] + (['xi_left', 'xi_right'] if free_tails else [])
            self.data.xnames = names
            self.k_extra = len(names) - self.exog.shape[1]
            self.df_model = float(len(names) - self.k_constant - self.k_extra)
            self.df_resid = float(np.shape(self.endog)[0] - len(names))

            return super().fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)

    _PHATFIT_CACHE.append(PhatFit)
    return PhatFit
