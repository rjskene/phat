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

PARAM_NAMES = ('mu', 'sig', 'xi_l', 'xi_r')
_XI_MAX = 10.0


@dataclass
class PhatFitResult:
    """
    Result of a maximum-likelihood fit of :class:`~phat.Phat`.

    Attributes
    ----------
    params:     estimates of the free parameters, in :attr:`names` order
    names:      names of the free parameters (``mu``, ``sig`` and, unless fixed, ``xi_l``, ``xi_r``)
    bse:        standard errors (square roots of the diagonal of :attr:`cov`); ``nan``
                where the observed information is not invertible, for instance when a
                tail shape sits on the ``xi = 0`` boundary
    cov:        inverse observed-information matrix
    llf:        log-likelihood at the optimum
    nobs:       number of observations
    fixed:      parameters held fixed during the fit (``xi_l``, ``xi_r`` and ``p``)
    converged:  whether the optimiser reported success
    message:    the optimiser's message
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

    @property
    def params_dict(self) -> dict[str, float]:
        """``{name: estimate}`` for the free parameters."""
        return {name: float(v) for name, v in zip(self.names, self.params)}

    @property
    def all_params(self) -> dict[str, float]:
        """``mu, sig, xi_l, xi_r`` with fixed values filled in."""
        out = dict(self.params_dict)
        for name in ('xi_l', 'xi_r'):
            if name in self.fixed:
                out[name] = float(self.fixed[name])
        return {name: out[name] for name in PARAM_NAMES}

    @property
    def dist(self) -> Phat:
        """The fitted distribution."""
        return Phat(**self.all_params, p=self.fixed.get('p'))

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
        lines.append(f'{"":6} {"estimate":>12} {"std err":>12}')
        for name, est, se in zip(self.names, self.params, self.bse):
            lines.append(f'{name:6} {est:12.6f} {se:12.6f}')
        for name, value in self.fixed.items():
            lines.append(f'{name:6} {str(value):>12} {"(fixed)":>12}')
        if not self.converged:
            lines.append(f'WARNING: optimiser did not converge: {self.message}')
        return '\n'.join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _start_values(x: np.ndarray, free_tails: bool) -> np.ndarray:
    """
    Robust starting point: the median for ``mu``; the interquartile range for
    ``sig`` (for ``xi`` around 0.2 the Phat IQR is about ``2.6 sig``); ``0.2``
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
    xi_l: float | None = None,
    xi_r: float | None = None,
    p: Sequence[float] | None = None,
    start: Sequence[float] | None = None,
    **minimize_kwargs: Any,
) -> PhatFitResult:
    """
    Maximum-likelihood fit of :class:`~phat.Phat` with ``scipy.optimize.minimize``.

    Parameters
    ----------
    values:     observations
    xi_l, xi_r: tail shapes to hold fixed. Give both for the two-stage estimator
                (shapes from the Hill double bootstrap, then ``mu`` and ``sig`` by
                likelihood); give neither for the full four-parameter fit.
    p:          mixture weights to hold fixed (default ``(0.5, 0.5)``)
    start:      starting values for the free parameters; see :func:`_start_values`
    **minimize_kwargs:  forwarded to ``scipy.optimize.minimize`` (default method
                ``L-BFGS-B`` with ``sig > 0`` and ``0 <= xi <= 10``)
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size < 3:
        raise ValueError('need at least three observations')
    if (xi_l is None) != (xi_r is None):
        raise ValueError('fix both tail shapes or neither')
    free_tails = xi_l is None

    def unpack(theta: np.ndarray) -> Phat:
        if free_tails:
            return Phat(theta[0], theta[1], theta[2], theta[3], p=p)
        assert xi_l is not None and xi_r is not None
        return Phat(theta[0], theta[1], float(xi_l), float(xi_r), p=p)

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
    H = _numerical_hessian(nll, theta)
    cov = np.full_like(H, np.nan)
    if np.all(np.isfinite(H)):
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            pass
    bse = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))

    fixed: dict[str, Any] = {}
    if not free_tails:
        fixed['xi_l'], fixed['xi_r'] = float(xi_l), float(xi_r)  # type: ignore[arg-type]
    if p is not None:
        fixed['p'] = tuple(float(v) for v in p)
    names = PARAM_NAMES if free_tails else PARAM_NAMES[:2]
    return PhatFitResult(
        params=theta, names=names, bse=bse, cov=cov, llf=-float(res.fun), nobs=int(x.size),
        fixed=fixed, converged=bool(res.success), message=str(res.message),
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

            names = ['mu', 'sig'] + (['xi_l', 'xi_r'] if free_tails else [])
            self.data.xnames = names
            self.k_extra = len(names) - self.exog.shape[1]
            self.df_model = float(len(names) - self.k_constant - self.k_extra)
            self.df_resid = float(np.shape(self.endog)[0] - len(names))

            return super().fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwds)

    _PHATFIT_CACHE.append(PhatFit)
    return PhatFit
