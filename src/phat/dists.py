"""
Houses all scipy-based distributions.

Separate code required for Tensorflow. Found in .learn.dists

WARNING: Only CarBen and Phat distributions should be used. The remaining
distributions are for illustration purposes in the docs.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import Any, Union

import numpy as np
import scipy.special as spec
import scipy.stats as scist

from phat._compat import AliasDict, renamed_attribute, renamed_kwargs
from phat.utils import XTYPE, argsetter, arrayarize, sech_squared


class BLineSig:
    """
    Borderline Sigmoid function defined by Taleb in SCOFT https://arxiv.org/abs/2001.10488

    FOR ILLUSTRATION PURPOSES ONLY
    """
    def __init__(self, k:float=1):
        self.k = k

    @argsetter('xk')
    def sf(self, x, k:float | None=None):
        return .5*(1 - np.tanh(k*x))

    @argsetter('xk')
    def pdf(self, x, k:float | None=None):
        return .5*k*sech_squared(x,k)  # type: ignore[operator]  # argsetter fills k

    @argsetter('k')
    def var(self, k):
        return np.pi**2 / (12*k**2)

    @argsetter('k')
    def m4(self, k):
        return 7*np.pi**4 / (240*k**4)

    @argsetter('k')
    def kurt(self, k):
        return self.m4(k) / self.var(k)**2

class LogPareto:
    """
    LogPareto as discussed in SCOFT https://arxiv.org/abs/2001.10488.

    FOR ILLUSTRATION PURPOSES ONLY
    """
    def __init__(self, alpha=2, L=1):
        self.alpha = alpha
        self.L = L

    def sf(self, x):
        return (self.L**self.alpha)*(np.log(x)**(-self.alpha))

class DBLFat(scist.rv_continuous):
    """
    Baseclass for group of twin-tailed fat-tailed distributions

    FOR ILLUSTRATION PURPOSES ONLY
    """
    @property
    def dist(self):
        raise NotImplementedError

    @property
    def qjunc(self):
        return .5

    @argsetter(['x', 'shape'])
    def _pdf(self, x, shape):
        pdf = np.where(
            x>=0,
            self.dist.pdf(x, shape),
            self.dist.pdf(-x, shape)
        )
        return pdf / 2

    @argsetter(['x', 'shape'])
    def _cdf(self, x, shape):
        cdf = np.where(
            x>=0,
            .5 + (self.dist.cdf(x, shape)/2),

            self.dist.sf(-x, shape)/2
        )
        return cdf

    @argsetter(['q', 'shape'])
    def _ppf(self, q, shape):
        if isinstance(q, (float, int)):
            q = np.array([q])

        ppf = np.where(
            q>=self.qjunc,
            self.dist.ppf(1/self.qjunc*(q - self.qjunc), shape),
            -self.dist.isf(q*1/self.qjunc, shape)
        )

        return ppf

    def _rvs(self, *args, **kwargs):
        rvs = super()._rvs(*args, **kwargs)
        return rvs

class DBLGP(DBLFat):
    dist = scist.genpareto

class DBLLomax(DBLFat):
    dist = scist.lomax

class DBLPareto(DBLFat):
    dist = scist.pareto

############# Main Phat-related Distributions ################

def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _scalar_or_array(out):
    """Return a NumPy scalar for 0-d results, otherwise the array (SciPy convention)."""
    out = np.asarray(out)
    return out[()] if out.ndim == 0 else out


def _rng(random_state=None) -> Union[np.random.Generator, np.random.RandomState]:
    """Coerce ``None``, an integer seed, a ``Generator`` or a legacy ``RandomState``."""
    if isinstance(random_state, (np.random.Generator, np.random.RandomState)):
        return random_state
    return np.random.default_rng(random_state)


class CarBenBase:
    """
    Base class of the CarBen hybrid distribution (Carreau and Bengio, 2007/2009),
    a one-sided splice of a Gaussian and a generalized Pareto distribution.
    Generalizes code used in both the left- and right-tailed versions.

    Three free parameters:

    * ``xi``:  shape of the generalized Pareto tail (``xi >= 0`` for a heavy tail)
    * ``mu``:  location of the Gaussian body
    * ``sigma``: scale of the Gaussian body. **Not** the standard deviation of the
      hybrid; see :meth:`std`.

    The two remaining GPD parameters are fixed by continuity of the density and
    its first derivative at the junction ``a``. With ``z = (1 + xi)^2 / (2 pi)``,
    ``s = sqrt(W(z))`` (``W`` the Lambert W function):

    * junction          ``a = mu + s * sigma``       (right tail; ``mu - s * sigma`` left)
    * GPD scale         ``b = sigma * (1 + xi) / s``
    * normaliser        ``gamma = 1 + Phi(s)``

    so the Gaussian part carries mass ``Phi(s) / gamma`` and the GPD tail
    ``1 / gamma``. Instead of ``sigma`` the junction ``a`` may be supplied.

    Attributes
    ----------
    body:   frozen ``scipy.stats.norm(mu, sigma)``
    tail:   frozen ``scipy.stats.genpareto`` (for the left tail: of the reflected variable)
    gamma:  normaliser; the density is (Gaussian or GPD) / gamma
    s:      junction distance from ``mu`` in units of ``sigma``
    """

    sig = renamed_attribute('sigma')

    @renamed_kwargs(sig='sigma')
    def __init__(self,
        xi: float,
        mu: float,
        sigma: float | None = None,
        a: float | None = None,
        ) -> None:
        xi = float(xi)
        if xi <= -1:
            raise ValueError('`xi` must be greater than -1')
        self.xi = xi
        self.mu = float(mu)

        provided = (sigma is None, a is None)
        if all(provided) or not any(provided):
            raise ValueError('Please provide only one of `sigma` or `a`')
        elif sigma is not None:
            sigma = float(sigma)
            if sigma <= 0:
                raise ValueError('`sigma` must be positive')
            self.sigma = sigma
            self._given = 'sigma'
            self.a = self._calc_a()
        else:
            assert a is not None
            self.a = float(a)
            self._given = 'a'
            self.sigma = self._calc_sigma()
            if not self.sigma > 0:
                raise ValueError('`a` must lie on the tail side of `mu`')
        self.b = self._calc_b_w_sig()

        self.body = scist.norm(self.mu, self.sigma)

    # ---- fixed quantities -------------------------------------------------
    @property
    def tail(self):
        """Frozen ``scipy.stats.genpareto``; assigned in the Left/Right subclass."""
        if hasattr(self, '_tail'):
            return self._tail
        raise NotImplementedError

    def z(self) -> float:
        """``(1 + xi)^2 / (2 pi)``, the argument of the Lambert W function."""
        return (1 + self.xi) ** 2 / (2 * np.pi)

    def W_z(self) -> float:
        """Principal branch of the Lambert W function at :meth:`z` (real for ``xi > -1``)."""
        val = spec.lambertw(self.z())
        return float(val.real)

    @property
    def s(self) -> float:
        """Junction distance from ``mu`` in units of ``sigma``: ``sqrt(W(z))``."""
        return float(np.sqrt(self.W_z()))

    @property
    def gamma(self) -> float:
        """Normaliser ``1 + Phi(s)``; equals ``1 + (1 + erf(s / sqrt 2)) / 2``."""
        return 1.0 + float(scist.norm.cdf(self.s))

    @property
    def body_mass(self) -> float:
        """Probability carried by the Gaussian part, ``Phi(s) / gamma``."""
        return float(scist.norm.cdf(self.s)) / self.gamma

    @property
    def tail_mass(self) -> float:
        """Probability carried by the GPD part, ``1 / gamma``."""
        return 1.0 / self.gamma

    def _calc_b_w_sig(self) -> float:
        """Equation 7 in Carreau and Bengio: ``b = sigma (1 + xi) / sqrt(W(z))``."""
        return self.sigma * (1 + self.xi) / self.s

    def _calc_b_w_a(self) -> float:
        """``b`` from the junction: ``sigma = |a - mu| / s`` so ``b = (1 + xi) |a - mu| / W(z)``."""
        return (1 + self.xi) * abs(self.a - self.mu) / self.W_z()

    # ---- defined by the right/left subclasses ----------------------------------
    def _calc_a(self) -> float:
        raise NotImplementedError

    def _calc_sigma(self) -> float:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def var(self) -> float:
        raise NotImplementedError

    def ppf(self, q: XTYPE) -> XTYPE:
        raise NotImplementedError

    # ---- generic helpers --------------------------------------------------
    def _zscore(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sigma

    def std(self) -> float:
        return float(np.sqrt(self.var()))

    def stats(self):
        """``(mean, var)``"""
        return self.mean(), self.var()

    def rvs(self,
        size: Union[int, Sequence[int], None] = None,
        random_state=None,
        *,
        seed=None,
        ) -> XTYPE:
        """
        Random draws by inversion of the CDF (exact for a single component).

        Parameters
        ----------
        size:           int or tuple of ints; ``None`` returns a scalar
        random_state:   ``None``, int seed, ``numpy.random.Generator`` or ``RandomState``
        seed:           deprecated alias of ``random_state``
        """
        rng = _rng(random_state if random_state is not None else seed)
        return self.ppf(rng.random(size))


class CarBenRight(CarBenBase):
    """
    Right-tailed CarBen hybrid: Gaussian for ``x <= a``, generalized Pareto for ``x > a``.
    """

    @renamed_kwargs(sig='sigma')
    def __init__(self,
        xi: float,
        mu: float,
        sigma: float | None = None,
        a: float | None = None,
        ) -> None:
        super().__init__(xi, mu, sigma, a)
        self._tail = scist.genpareto(self.xi, loc=self.a, scale=self.b)
        self._phi_s = float(scist.norm.cdf(self.s))
        self._log_gamma = float(np.log(self.gamma))

    def _calc_a(self) -> float:
        """Equation 8 in Carreau and Bengio: ``a = mu + sigma sqrt(W(z))``."""
        return self.mu + self.sigma * self.s

    def _calc_sigma(self) -> float:
        return (self.a - self.mu) / self.s

    # ---- distribution functions ---------------------------------------------
    def pdf(self, x: XTYPE) -> np.ndarray:
        x = _as_float_array(x)
        out = np.where(x > self.a, self.tail.pdf(x), self.body.pdf(x)) / self.gamma
        return _scalar_or_array(out)

    def logpdf(self, x: XTYPE) -> np.ndarray:
        x = _as_float_array(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(x > self.a, self.tail.logpdf(x), self.body.logpdf(x)) - self._log_gamma
        return _scalar_or_array(out)

    def cdf(self, x: XTYPE) -> np.ndarray:
        x = _as_float_array(x)
        out = np.where(x > self.a, self._phi_s + self.tail.cdf(x), self.body.cdf(x)) / self.gamma
        return _scalar_or_array(out)

    def sf(self, x: XTYPE) -> np.ndarray:
        """Survival function ``P(X > x)``, computed without ``1 - cdf`` so the far tail keeps precision."""
        x = _as_float_array(x)
        out = np.where(x > self.a, self.tail.sf(x), 1.0 + self._phi_s - self.body.cdf(x)) / self.gamma
        return _scalar_or_array(out)

    def logcdf(self, x: XTYPE) -> np.ndarray:
        x = _as_float_array(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(
                x > self.a,
                np.log(self._phi_s + self.tail.cdf(x)),
                self.body.logcdf(x),
            ) - self._log_gamma
        return _scalar_or_array(out)

    def logsf(self, x: XTYPE) -> np.ndarray:
        x = _as_float_array(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(
                x > self.a,
                self.tail.logsf(x),
                np.log1p(self._phi_s - self.body.cdf(x)),
            ) - self._log_gamma
        return _scalar_or_array(out)

    @property
    def qjunc(self) -> float:
        """CDF at the junction ``a``: ``1 - 1 / gamma``."""
        return 1 - 1 / self.gamma

    def ppf(self, q: XTYPE) -> np.ndarray:
        """Quantile function. Uses the GPD ``isf`` above the junction so that ``q`` near 1 stays precise."""
        q = _as_float_array(q)
        with np.errstate(invalid='ignore'):
            out = np.where(
                q > self.qjunc,
                self.tail.isf(self.gamma * (1.0 - q)),
                self.body.ppf(self.gamma * q),
            )
        return _scalar_or_array(out)

    def isf(self, p: XTYPE) -> np.ndarray:
        """Inverse survival function: ``x`` with ``P(X > x) = p``."""
        p = _as_float_array(p)
        with np.errstate(invalid='ignore'):
            out = np.where(
                p < self.tail_mass,
                self.tail.isf(self.gamma * p),
                # body.isf(gamma*p - Phi(s)) rewritten so p = 1 gives exactly -inf
                self.body.ppf(self.gamma * (1.0 - p)),
            )
        return _scalar_or_array(out)

    # ---- moments ------------------------------------------------------------
    def _gpd_mean(self) -> float:
        return self.a + self.b / (1 - self.xi) if self.xi < 1 else np.inf

    def _gpd_var(self) -> float:
        if self.xi >= 0.5:
            return np.inf
        return self.b ** 2 / ((1 - self.xi) ** 2 * (1 - 2 * self.xi))

    def _truncated_body(self):
        """The Gaussian body conditioned on ``x <= a`` (bounds standardised, as SciPy requires)."""
        return scist.truncnorm(-np.inf, self.s, loc=self.mu, scale=self.sigma)

    def mean(self) -> float:
        """First moment; ``inf`` for ``xi >= 1``."""
        if self.xi >= 1:
            return np.inf
        return self.body_mass * float(self._truncated_body().mean()) + self.tail_mass * self._gpd_mean()

    def var(self) -> float:
        """Variance; ``inf`` for ``xi >= 1/2``."""
        if self.xi >= 0.5:
            return np.inf
        tb = self._truncated_body()
        m_b, v_b = float(tb.mean()), float(tb.var())
        m_t, v_t = self._gpd_mean(), self._gpd_var()
        second = self.body_mass * (v_b + m_b ** 2) + self.tail_mass * (v_t + m_t ** 2)
        return second - self.mean() ** 2

    # ---- partial first moments (for expected shortfall) ---------------------
    def _gauss_lower_partial(self, t: np.ndarray) -> np.ndarray:
        """``int_{-inf}^{min(t, a)} x phi(x) dx`` (un-normalised Gaussian part)."""
        z = np.minimum(self._zscore(t), self.s)
        return self.mu * scist.norm.cdf(z) - self.sigma * scist.norm.pdf(z)

    def _gauss_upper_partial(self, t: np.ndarray) -> np.ndarray:
        """``int_{min(t, a)}^{a} x phi(x) dx``."""
        z = np.minimum(self._zscore(t), self.s)
        body = self._phi_s - scist.norm.cdf(z)
        return self.mu * body - self.sigma * (scist.norm.pdf(self.s) - scist.norm.pdf(z))

    def _gpd_lower_partial(self, t: np.ndarray) -> np.ndarray:
        """``int_{a}^{max(t, a)} x g(x) dx`` for the un-normalised GPD part; finite for every ``xi``."""
        y = np.maximum(t - self.a, 0.0)
        sf = self.tail.sf(np.maximum(t, self.a))
        g = self.tail.cdf(np.maximum(t, self.a))
        if self.xi == 1:
            inner = -y * sf + self.b * np.log1p(y / self.b)
        else:
            inner = -y * sf + (self.b / (1 - self.xi)) * (1.0 - (1.0 + self.xi * y / self.b) * sf)
        return self.a * g + inner

    def _gpd_upper_partial(self, t: np.ndarray) -> np.ndarray:
        """``int_{max(t, a)}^{inf} x g(x) dx``; ``inf`` for ``xi >= 1``."""
        t = np.asarray(t, float)
        if self.xi >= 1:
            return np.full(t.shape, np.inf)
        t_ = np.maximum(t, self.a)
        sf = self.tail.sf(t_)
        return sf * (t_ + (self.b + self.xi * (t_ - self.a)) / (1 - self.xi))

    def lower_partial_mean(self, t: XTYPE) -> np.ndarray:
        """``int_{-inf}^{t} x f(x) dx``."""
        t = _as_float_array(t)
        return _scalar_or_array((self._gauss_lower_partial(t) + self._gpd_lower_partial(t)) / self.gamma)

    def upper_partial_mean(self, t: XTYPE) -> np.ndarray:
        """``int_{t}^{inf} x f(x) dx``; ``inf`` for ``xi >= 1``."""
        t = _as_float_array(t)
        return _scalar_or_array((self._gauss_upper_partial(t) + self._gpd_upper_partial(t)) / self.gamma)


class CarBenLeft(CarBenBase):
    """
    Left-tailed CarBen hybrid: the reflection of :class:`CarBenRight` about ``mu``,
    ``f_L(x) = f_R(2 mu - x)``. Generalized Pareto for ``x < a``, Gaussian for ``x >= a``.

    Every method is evaluated through the mirrored right-tailed component, so the
    two sides cannot drift apart.
    """

    @renamed_kwargs(sig='sigma')
    def __init__(self,
        xi: float,
        mu: float,
        sigma: float | None = None,
        a: float | None = None,
        ) -> None:
        super().__init__(xi, mu, sigma, a)
        self._mirror = CarBenRight(self.xi, self.mu, sigma=self.sigma)
        # Tail of the reflected variable -x, kept for backwards compatibility:
        # ``tail.pdf(-x)`` is the GPD density of the left tail at ``x``.
        self._tail = scist.genpareto(self.xi, loc=-self.a, scale=self.b)

    def _calc_a(self) -> float:
        """Equation 8 with the sign of the second term flipped: ``a = mu - sigma sqrt(W(z))``."""
        return self.mu - self.sigma * self.s

    def _calc_sigma(self) -> float:
        return (self.mu - self.a) / self.s

    def _reflect(self, x: XTYPE) -> np.ndarray:
        return 2.0 * self.mu - _as_float_array(x)

    def pdf(self, x: XTYPE) -> np.ndarray:
        return self._mirror.pdf(self._reflect(x))

    def logpdf(self, x: XTYPE) -> np.ndarray:
        return self._mirror.logpdf(self._reflect(x))

    def cdf(self, x: XTYPE) -> np.ndarray:
        return self._mirror.sf(self._reflect(x))

    def sf(self, x: XTYPE) -> np.ndarray:
        return self._mirror.cdf(self._reflect(x))

    def logcdf(self, x: XTYPE) -> np.ndarray:
        return self._mirror.logsf(self._reflect(x))

    def logsf(self, x: XTYPE) -> np.ndarray:
        return self._mirror.logcdf(self._reflect(x))

    @property
    def qjunc(self) -> float:
        """CDF at the junction ``a``: ``1 / gamma``."""
        return 1 / self.gamma

    def ppf(self, q: XTYPE) -> np.ndarray:
        return _scalar_or_array(2.0 * self.mu - self._mirror.isf(q))

    def isf(self, p: XTYPE) -> np.ndarray:
        return _scalar_or_array(2.0 * self.mu - self._mirror.ppf(p))

    def mean(self) -> float:
        """First moment; ``-inf`` for ``xi >= 1``."""
        return 2.0 * self.mu - self._mirror.mean()

    def var(self) -> float:
        return self._mirror.var()

    def lower_partial_mean(self, t: XTYPE) -> np.ndarray:
        """``int_{-inf}^{t} x f(x) dx``; ``-inf`` for ``xi >= 1``."""
        r = self._reflect(t)
        return _scalar_or_array(2.0 * self.mu * self._mirror.sf(r) - self._mirror.upper_partial_mean(r))

    def upper_partial_mean(self, t: XTYPE) -> np.ndarray:
        """``int_{t}^{inf} x f(x) dx``."""
        r = self._reflect(t)
        return _scalar_or_array(2.0 * self.mu * self._mirror.cdf(r) - self._mirror.lower_partial_mean(r))


_UNSET: Any = object()


class CarBenHybrid:
    """
    Factory that returns a :class:`CarBenRight` or a :class:`CarBenLeft`.

    Parameters
    ----------
    xi:     shape of the generalized Pareto tail, ``xi > -1``: positive for a
            heavy (power-law) tail, zero for an exponential tail, negative for
            a bounded tail that ends at ``a + b / |xi|`` (``a - b / |xi|`` on
            the left). The sign of ``xi`` says nothing about the side.
    mu:     location of the Gaussian body
    sigma:  scale of the Gaussian body (or give the junction ``a`` instead)
    a:      junction between body and tail
    tail:   ``'right'`` (default) or ``'left'``

    .. versionchanged:: 0.1
       ``xi`` is always the tail shape. In 0.0.x a negative ``xi`` meant "left
       tail with shape ``|xi|``". For one release that reading is kept, with a
       ``FutureWarning``, when ``xi < 0`` is passed **without** a side; pass
       ``tail='left'`` and a positive ``xi`` instead. With a side given, a
       negative ``xi`` is already a bounded tail on that side.

    .. deprecated:: 0.1
       ``rtail=True|False`` (use ``tail='right'|'left'``) and ``sig`` (use
       ``sigma``).
    """

    @renamed_kwargs(sig='sigma')
    def __new__(cls,
        xi: float,
        mu: float,
        sigma: float | None = None,
        a: float | None = None,
        tail: str = _UNSET,
        *,
        rtail: bool = _UNSET,
        ) -> Any:
        xi = float(xi)
        if isinstance(tail, (bool, np.bool_)):       # 0.0.x: rtail as fifth positional argument
            if rtail is not _UNSET:
                raise TypeError('CarBenHybrid() got both `tail` and `rtail` (deprecated)')
            tail, rtail = _UNSET, bool(tail)
        if rtail is not _UNSET:
            if tail is not _UNSET:
                raise TypeError('CarBenHybrid() got both `tail` and `rtail` (deprecated)')
            tail = 'right' if rtail else 'left'
            warnings.warn(
                f"`rtail` is deprecated; use `tail={tail!r}`. "
                'The old name will be removed in a future release.',
                DeprecationWarning,
                stacklevel=3,
            )

        if tail is _UNSET:
            tail = 'right'
            if xi < 0:
                warnings.warn(
                    'CarBenHybrid(xi < 0) without `tail` builds a left tail with shape |xi|, '
                    'as in phat-tails 0.0.x. From the next release `xi` is always the tail shape '
                    '(negative = bounded tail) and the default side is the right. '
                    "Use CarBenHybrid(abs(xi), ..., tail='left') or CarBenLeft(abs(xi), ...).",
                    FutureWarning,
                    stacklevel=3,
                )
                xi, tail = -xi, 'left'
        if tail not in ('left', 'right'):
            raise ValueError("`tail` must be 'left' or 'right'")

        return CarBenRight(xi, mu, sigma, a) if tail == 'right' else CarBenLeft(xi, mu, sigma, a)


class Phat:
    """
    Two-tailed distribution: a mixture of a left-tailed CarBen hybrid and a
    right-tailed CarBen hybrid that share the Gaussian location and scale.

    Parameters
    ----------
    mu:         location of the shared Gaussian body
    sigma:      scale of the shared Gaussian body. **Not** the standard deviation
                of the Phat variable; see :meth:`std`. For
                ``xi_left = xi_right = 0.2`` the standard deviation is about
                ``4.6 sigma``.
    xi_left:    shape of the left generalized Pareto tail, ``> -1``. Positive for
                a heavy (power-law) tail, the case Phat is built for; zero for an
                exponential tail; negative for a bounded GPD tail.
    xi_right:   shape of the right generalized Pareto tail, same convention. Each
                shape describes its own side: a fat left tail has ``xi_left > 0``.
    weights:    mixture weights ``(p_l, p_r)``, non-negative and summing to 1.
                Default ``(0.5, 0.5)``.

    .. deprecated:: 0.1
       The 0.0.x names ``sig``, ``xi_l``, ``xi_r`` and ``p`` still work as
       keywords and attributes and emit a ``DeprecationWarning``.

    Attributes
    ----------
    left:   :class:`CarBenLeft`
    right:  :class:`CarBenRight`

    Notes
    -----
    Density ``f = p_l f_L(x; xi_left) + p_r f_R(x; xi_right)``; the CDF is the same
    weighted sum. The body between the two junctions ``a_left < a_right`` is Gaussian,
    but it carries only 18-28% of the mass for ``xi`` between 0 and 1 (see
    :attr:`body_mass`); outside a junction the density is a GPD tail plus the
    Gaussian remnant of the other component. At ``xi_left = xi_right = 0`` the tails are
    exponential, so Phat is not Gaussian.

    With a negative shape the GPD tail of that component ends at
    ``a_right + b_right / |xi_right|`` (``a_left - b_left / |xi_left|`` on the
    left). The mixture is still supported on the whole real line, because the
    other component's
    Gaussian body extends past that endpoint; beyond it the density is that
    Gaussian remnant alone.

    The mean exists for ``max(xi_left, xi_right) < 1`` and the variance for
    ``max(xi_left, xi_right) < 1/2``; beyond those thresholds :meth:`mean` returns
    ``inf``/``-inf``/``nan`` and :meth:`var` returns ``inf``.

    The mixing of a hybrid Pareto with its reflection was proposed and used by
    Carreau (2007, PhD thesis, Universite de Montreal, sections 3.2 and 4.5).
    """
    PARAM_NAMES = ['mu', 'sigma', 'xi_left', 'xi_right', 'a_left', 'a_right', 'b_left', 'b_right']

    _OLD_PARAM_NAMES = {
        'sig': 'sigma', 'xi_l': 'xi_left', 'xi_r': 'xi_right',
        'a_l': 'a_left', 'a_r': 'a_right', 'b_l': 'b_left', 'b_r': 'b_right',
    }

    # 0.0.x attribute names
    sig = renamed_attribute('sigma')
    xi_l = renamed_attribute('xi_left')
    xi_r = renamed_attribute('xi_right')
    p = renamed_attribute('weights')

    @renamed_kwargs(sig='sigma', xi_l='xi_left', xi_r='xi_right', p='weights')
    def __init__(
        self,
        mu: float,
        sigma: float,
        xi_left: float,
        xi_right: float,
        weights: Iterable[float] | None = None,
        ) -> None:
        mu, sigma, xi_left, xi_right = (float(v) for v in (mu, sigma, xi_left, xi_right))
        if not sigma > 0:
            raise ValueError('`sigma` must be positive')
        if xi_left <= -1 or xi_right <= -1:
            raise ValueError('`xi_left` and `xi_right` must be greater than -1')
        self.mu = mu
        self.sigma = sigma
        self.xi_left = xi_left
        self.xi_right = xi_right

        self.left = CarBenLeft(xi_left, mu, sigma)
        self.right = CarBenRight(xi_right, mu, sigma)

        w = np.array([.5, .5]) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (2,) or np.any(w < 0) or not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError('`weights` must be two non-negative weights summing to 1')
        self.weights = w

    # ---- parameters ---------------------------------------------------------
    @property
    def args(self) -> tuple:
        """Tuple of the distribution parameters, in :attr:`PARAM_NAMES` order."""
        return self.mu, self.sigma, self.xi_left, self.xi_right, self.left.a, \
            self.right.a, self.left.b, self.right.b

    @property
    def params(self) -> dict:
        """Dictionary of the distribution parameters."""
        return AliasDict(dict(zip(self.PARAM_NAMES, self.args)), self._OLD_PARAM_NAMES)

    @property
    def learnable_params(self) -> list[float]:
        """The four free parameters ``(mu, sigma, xi_left, xi_right)``."""
        return [self.params[name] for name in self.PARAM_NAMES[:4]]

    @property
    def junctions(self) -> tuple:
        """``(a_left, a_right)``: the points where the Gaussian body meets each GPD tail."""
        return self.left.a, self.right.a

    @property
    def body_mass(self) -> float:
        """``P(a_left < X < a_right)``, the probability carried by the Gaussian body."""
        a_left, a_right = self.junctions
        return float(self.cdf(a_right) - self.cdf(a_left))

    @property
    def tail_masses(self) -> tuple:
        """``(P(X < a_left), P(X > a_right))``."""
        a_left, a_right = self.junctions
        return float(self.cdf(a_left)), float(self.sf(a_right))

    def _components(self):
        return (self.left, self.right)

    def _weighted(self, values) -> float:
        """``sum(p_i * v_i)`` skipping zero weights, so ``0 * inf`` does not poison the result."""
        with np.errstate(invalid='ignore'):
            return float(sum(w * v for w, v in zip(self.weights, values) if w > 0))

    def _mix(self, name: str, x) -> np.ndarray:
        x = _as_float_array(x)
        out = self.weights[0] * getattr(self.left, name)(x) + self.weights[1] * getattr(self.right, name)(x)
        return _scalar_or_array(out)

    def _logmix(self, name: str, x) -> np.ndarray:
        x = _as_float_array(x)
        with np.errstate(divide='ignore'):
            logp = np.log(self.weights)
        out = np.logaddexp(
            logp[0] + getattr(self.left, name)(x),
            logp[1] + getattr(self.right, name)(x),
        )
        return _scalar_or_array(out)

    # ---- distribution functions ---------------------------------------------
    def pdf(self, x: XTYPE) -> np.ndarray:
        """Probability density function."""
        return self._mix('pdf', x)

    def logpdf(self, x: XTYPE) -> np.ndarray:
        """Log of the density, finite far into either tail."""
        return self._logmix('logpdf', x)

    def cdf(self, x: XTYPE) -> np.ndarray:
        """Cumulative distribution function ``P(X <= x)``."""
        return self._mix('cdf', x)

    def logcdf(self, x: XTYPE) -> np.ndarray:
        return self._logmix('logcdf', x)

    def sf(self, x: XTYPE) -> np.ndarray:
        """Survival function ``P(X > x)``, accurate in the far right tail (not ``1 - cdf``)."""
        return self._mix('sf', x)

    def logsf(self, x: XTYPE) -> np.ndarray:
        return self._logmix('logsf', x)

    def _quantile(self, q: np.ndarray, upper: bool) -> np.ndarray:
        """
        Root of the mixture CDF (``upper=False``: ``cdf(x) = q``) or survival
        function (``upper=True``: ``sf(x) = q``), vectorised.

        For every ``q`` the root lies between the two component quantiles, which
        gives a guaranteed bracket. Safeguarded Newton steps (falling back to
        bisection) converge in a handful of iterations. Whichever of ``cdf`` and
        ``sf`` is the smaller probability is the one rooted, so both tails keep
        full relative precision.
        """
        q = np.asarray(q, dtype=float)
        out = np.full(q.shape, np.nan)
        out[q == 0] = np.inf if upper else -np.inf
        out[q == 1] = -np.inf if upper else np.inf
        interior = (q > 0) & (q < 1)
        if not interior.any():
            return out

        qi = q[interior]
        lower_p = 1.0 - qi if upper else qi          # P(X <= x) at the target
        upper_p = qi if upper else 1.0 - qi          # P(X > x) at the target
        use_sf = lower_p > 0.5

        if upper:
            x_l, x_r = self.left.isf(qi), self.right.isf(qi)
        else:
            x_l, x_r = self.left.ppf(qi), self.right.ppf(qi)
        lo = np.minimum(x_l, x_r)
        hi = np.maximum(x_l, x_r)

        def g(x):
            # increasing in x, zero at the quantile
            return np.where(use_sf, upper_p - self.sf(x), self.cdf(x) - lower_p)

        x = 0.5 * (lo + hi)
        rel_tol = 1e-13
        for _ in range(200):
            gx = g(x)
            below = gx < 0
            lo = np.where(below, x, lo)
            hi = np.where(below, hi, x)
            with np.errstate(divide='ignore', invalid='ignore'):
                newton = x - gx / self.pdf(x)
            ok = np.isfinite(newton) & (newton > lo) & (newton < hi)
            x_new = np.where(ok, newton, 0.5 * (lo + hi))
            done = (gx == 0) | (np.abs(x_new - x) <= rel_tol * np.maximum(1.0, np.abs(x)))
            x = np.where(gx == 0, x, x_new)
            if done.all():
                break

        out[interior] = x
        return out

    def ppf(self, q: XTYPE) -> np.ndarray:
        """
        Quantile function: ``x`` with ``P(X <= x) = q``.

        The quantile of a mixture is the root of the mixture CDF, not the
        weighted average of the component quantiles (which 0.0.x returned).
        """
        return _scalar_or_array(self._quantile(_as_float_array(q), upper=False))

    def isf(self, p: XTYPE) -> np.ndarray:
        """Inverse survival function: ``x`` with ``P(X > x) = p``."""
        return _scalar_or_array(self._quantile(_as_float_array(p), upper=True))

    def loglike(self, x: XTYPE) -> np.ndarray:
        """Log-likelihood per observation (alias of :meth:`logpdf`)."""
        return self.logpdf(x)

    def nll(self, x: XTYPE) -> np.ndarray:
        """Negative log-likelihood per observation."""
        return -self.logpdf(x)

    # ---- sampling -----------------------------------------------------------
    def rvs(self,
        size: Union[int, Iterable[int]] = None,
        random_state=None,
        *,
        seed=None,
        ) -> XTYPE:
        """
        Exact random draws: pick a component with probability ``weights``, then invert
        that component's CDF. (0.0.x pushed uniforms through the averaged
        ``ppf``, which does not sample the stated density.)

        Parameters
        ----------
        size:           int or tuple of ints; ``None`` returns a scalar
        random_state:   ``None``, int seed, ``numpy.random.Generator`` or ``RandomState``
        seed:           alias of ``random_state`` kept for 0.0.x callers
        """
        rng = _rng(random_state if random_state is not None else seed)
        shape = () if size is None else tuple(np.atleast_1d(size).astype(int))
        u = rng.random(shape)
        from_left = rng.random(shape) < self.weights[0]
        x = np.where(from_left, self.left.ppf(u), self.right.ppf(u))
        return _scalar_or_array(x)

    def std_rvs(self, *args, **kwargs) -> XTYPE:
        """Draws divided by the standard deviation, so the sample has unit scale. Requires ``xi < 1/2``."""
        sd = self.std()
        if not np.isfinite(sd):
            raise ValueError('standard deviation is infinite: both tail shapes must be below 1/2')
        return np.asarray(self.rvs(*args, **kwargs)) / sd

    # ---- moments ------------------------------------------------------------
    def mean(self) -> float:
        """Mean. ``inf`` / ``-inf`` when only the right / left tail has ``xi >= 1``, ``nan`` when both do."""
        return self._weighted([c.mean() for c in self._components()])

    def var(self) -> float:
        """Variance ``sum p_i (v_i + m_i^2) - mean^2``; ``inf`` when any weighted tail has ``xi >= 1/2``."""
        comps = [(w, c) for w, c in zip(self.weights, self._components()) if w > 0]
        if any(c.xi >= 0.5 for _, c in comps):
            return np.inf
        second = sum(w * (c.var() + c.mean() ** 2) for w, c in comps)
        return float(second - self.mean() ** 2)

    def std(self) -> float:
        """Standard deviation; ``inf`` when the variance is."""
        return float(np.sqrt(self.var()))

    def stats(self) -> tuple:
        """``(mean, var)``"""
        return self.mean(), self.var()

    def expected_shortfall(self, q: XTYPE, tail: str = 'left') -> np.ndarray:
        """
        Expected shortfall at tail probability ``q``.

        ``tail='left'``:  ``E[X | X <= ppf(q)]``, the usual loss-side measure for returns.
        ``tail='right'``: ``E[X | X >= isf(q)]``; ``inf`` when ``xi_right >= 1``.
        """
        q = _as_float_array(q)
        if tail == 'left':
            x = self._quantile(q, upper=False)
            partial = [c.lower_partial_mean(x) for c in self._components()]
        elif tail == 'right':
            x = self._quantile(q, upper=True)
            partial = [c.upper_partial_mean(x) for c in self._components()]
        else:
            raise ValueError("`tail` must be 'left' or 'right'")
        with np.errstate(invalid='ignore', divide='ignore'):
            num = sum(w * np.asarray(v, float) for w, v in zip(self.weights, partial) if w > 0)
            out = np.where((q > 0) & (q <= 1), num / q, np.nan)
        return _scalar_or_array(out)

    es = expected_shortfall

    # ---- estimation ---------------------------------------------------------
    @staticmethod
    @renamed_kwargs(xi_l='xi_left', xi_r='xi_right')
    def fit(
        values: Iterable[float],
        xi_left: float | None = None,
        xi_right: float | None = None,
        weights: Sequence[float] | None = None,
        *,
        backend: str = 'scipy',
        **kwargs: Any,
        ) -> Any:
        """
        Maximum-likelihood fit.

        With ``xi_left`` and ``xi_right`` given only ``(mu, sigma)`` are estimated (the
        two-stage estimator: tail shapes from
        :func:`~phat.bootstrap.two_tailed_hill_double_bootstrap`, body by
        likelihood); otherwise all four parameters are. ``weights`` fixes the mixture
        weights (default ``(0.5, 0.5)``). Both variants report standard errors.

        Parameters
        ----------
        backend:    ``'scipy'`` (default; NumPy + SciPy only) returns a
                    :class:`phat.fit.PhatFitResult`; ``'statsmodels'`` returns the
                    ``statsmodels`` results object of :class:`phat.fit.PhatFit`
                    and needs the ``fit`` extra.
        **kwargs:   forwarded to :func:`phat.fit.fit_phat` or to ``PhatFit.fit``
        """
        if backend == 'scipy':
            from phat.fit import fit_phat
            return fit_phat(values, xi_left, xi_right, weights=weights, **kwargs)
        if backend == 'statsmodels':
            if weights is not None:
                raise ValueError('the statsmodels backend does not take `weights`')
            from phat.fit import _statsmodels_phatfit
            x = np.asarray(values, dtype=float)
            exog = np.ones((x.size, 1))  # constant column; the model has no covariates
            return _statsmodels_phatfit()(x, exog, xi_left, xi_right).fit(**kwargs)
        raise ValueError(f"backend must be 'scipy' or 'statsmodels', not {backend!r}")

class PhatStack(Phat):
    """
    Weighted collection of Phat distributions used to generate a 2-d array of
    draws in which every row comes from a single component.

    This is NOT a mixture model. Along the first axis the rows are divided
    between the components in proportion to ``weights``; along the second axis each
    row is drawn from one component only, so a single regime governs each
    simulated path.

    Parameters
    ----------
    phats:  Phat distributions (as separate arguments or one iterable)
    weights: one weight per distribution, summing to 1

    Attributes
    ----------
    n_comps:    number of component distributions
    """

    @renamed_kwargs(p='weights')
    def __init__(
        self,
        *phats: Any,
        weights: Sequence[float] | None = None
        ) -> None:
        nested = len(phats) == 1 and not isinstance(phats[0], Phat)
        comps: tuple[Phat, ...] = tuple(phats[0]) if nested else phats

        if weights is None or len(comps) != len(weights):
            txt = 'Length of `weights` does not match number'
            txt += f' of distributions: {len(comps)} v {None if weights is None else len(weights)}'
            raise AssertionError(txt)

        self.n_comps = len(comps)
        self.phats = comps
        self.weights = arrayarize(weights)

    def rvs(self,  # type: ignore[override]  # PhatStack draws 2-d blocks, see the docstring
        size: Sequence[int] | None = None,
        random_state=None,
        return_splits: bool = False,
        *,
        seed=None,
        ) -> Any:
        """
        Array of random draws, shape ``size = (m, n)`` only.

        Rows are allocated to components as ``floor(m * weights)``; remainders go to
        the first components. Each component's rows are drawn with its own
        exact :meth:`Phat.rvs` from one shared random generator.

        Parameters
        ----------
        size:           ``(m, n)``
        random_state:   ``None``, int seed, ``numpy.random.Generator`` or ``RandomState``
        return_splits:  also return the number of rows drawn from each component
        seed:           alias of ``random_state`` kept for 0.0.x callers
        """
        if size is None or len(size) != 2:
            raise AssertionError('Size must be 2D m x n array')
        rng = _rng(random_state if random_state is not None else seed)

        iters = np.floor(size[0] * self.weights).astype(int)
        rmdr = size[0] - iters.sum()
        iters[:rmdr] += 1

        Y = np.zeros(tuple(size))
        start = 0
        for phat, end in zip(self.phats, iters):
            if end:
                Y[start:start + end] = phat.rvs((end, size[1]), random_state=rng)
            start += end

        if return_splits:
            return Y, iters
        return Y


def __getattr__(name):
    """``phat.dists.PhatFit`` moved to :mod:`phat.fit`; resolve it lazily for old imports."""
    if name == 'PhatFit':
        from phat.fit import PhatFit
        return PhatFit
    raise AttributeError(f"module 'phat.dists' has no attribute {name!r}")
