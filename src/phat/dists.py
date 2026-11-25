"""
Houses all scipy-based distributions.

Separate code required for Tensorflow. Found in .learn.dists

WARNING: Only CarBen and Phat distributions should be used. The remaining
distributions are for illustration purposes in the docs.
"""
from typing import Union, Iterable, Sequence, Callable, List
import numpy as np
import scipy.stats as scist
import scipy.special as spec
from scipy._lib._util import check_random_state
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

from phat.utils import argsetter, arrayarize, sech_squared, stacker, dotweight, XTYPE

class BLineSig:
    """
    Borderline Sigmoid function defined by Taleb in SCOFT https://arxiv.org/abs/2001.10488

    FOR ILLUSTRATION PURPOSES ONLY
    """
    def __init__(self, k:float=1):
        self.k = k

    @argsetter('xk')
    def sf(self, x, k:float=None):
        return .5*(1 - np.tanh(k*x))

    @argsetter('xk')
    def pdf(self, x, k:float=None):
        return .5*k*sech_squared(x,k)
    
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


def _rng(random_state=None) -> np.random.Generator:
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
    * ``sig``: scale of the Gaussian body. **Not** the standard deviation of the
      hybrid; see :meth:`std`.

    The two remaining GPD parameters are fixed by continuity of the density and
    its first derivative at the junction ``a``. With ``z = (1 + xi)^2 / (2 pi)``,
    ``s = sqrt(W(z))`` (``W`` the Lambert W function):

    * junction          ``a = mu + s * sig``       (right tail; ``mu - s * sig`` left)
    * GPD scale         ``b = sig * (1 + xi) / s``
    * normaliser        ``gamma = 1 + Phi(s)``

    so the Gaussian part carries mass ``Phi(s) / gamma`` and the GPD tail
    ``1 / gamma``. Instead of ``sig`` the junction ``a`` may be supplied.

    Attributes
    ----------
    body:   frozen ``scipy.stats.norm(mu, sig)``
    tail:   frozen ``scipy.stats.genpareto`` (for the left tail: of the reflected variable)
    gamma:  normaliser; the density is (Gaussian or GPD) / gamma
    s:      junction distance from ``mu`` in units of ``sig``
    """

    def __init__(self,
        xi: float,
        mu: float,
        sig: float = None,
        a: float = None,
        ) -> None:
        xi = float(xi)
        if xi <= -1:
            raise ValueError('`xi` must be greater than -1')
        self.xi = xi
        self.mu = float(mu)

        provided = (sig is None, a is None)
        if all(provided) or not any(provided):
            raise ValueError('Please provide only one of `sig` or `a`')
        elif sig is not None:
            sig = float(sig)
            if sig <= 0:
                raise ValueError('`sig` must be positive')
            self.sig = sig
            self._given = 'sig'
            self.a = self._calc_a()
        else:
            self.a = float(a)
            self._given = 'a'
            self.sig = self._calc_sig()
            if not self.sig > 0:
                raise ValueError('`a` must lie on the tail side of `mu`')
        self.b = self._calc_b_w_sig()

        self.body = scist.norm(self.mu, self.sig)

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
        """Junction distance from ``mu`` in units of ``sig``: ``sqrt(W(z))``."""
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
        """Equation 7 in Carreau and Bengio: ``b = sig (1 + xi) / sqrt(W(z))``."""
        return self.sig * (1 + self.xi) / self.s

    def _calc_b_w_a(self) -> float:
        """``b`` from the junction: ``sig = |a - mu| / s`` so ``b = (1 + xi) |a - mu| / W(z)``."""
        return (1 + self.xi) * abs(self.a - self.mu) / self.W_z()

    # ---- generic helpers --------------------------------------------------
    def _zscore(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / self.sig

    def std(self) -> float:
        return float(np.sqrt(self.var()))

    def stats(self):
        """``(mean, var)``"""
        return self.mean(), self.var()

    def rvs(self,
        size: Union[int, Iterable[int]] = None,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tail = scist.genpareto(self.xi, loc=self.a, scale=self.b)
        self._phi_s = float(scist.norm.cdf(self.s))
        self._log_gamma = float(np.log(self.gamma))

    def _calc_a(self) -> float:
        """Equation 8 in Carreau and Bengio: ``a = mu + sig sqrt(W(z))``."""
        return self.mu + self.sig * self.s

    def _calc_sig(self) -> float:
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
        """Survival function ``P(X > x)``, computed without ``1 - cdf`` so the far tail keeps its precision."""
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
                self.body.isf(self.gamma * p - self._phi_s),
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
        return scist.truncnorm(-np.inf, self.s, loc=self.mu, scale=self.sig)

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
        return self.mu * scist.norm.cdf(z) - self.sig * scist.norm.pdf(z)

    def _gauss_upper_partial(self, t: np.ndarray) -> np.ndarray:
        """``int_{min(t, a)}^{a} x phi(x) dx``."""
        z = np.minimum(self._zscore(t), self.s)
        return self.mu * (self._phi_s - scist.norm.cdf(z)) - self.sig * (scist.norm.pdf(self.s) - scist.norm.pdf(z))

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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mirror = CarBenRight(self.xi, self.mu, sig=self.sig)
        # Tail of the reflected variable -x, kept for backwards compatibility:
        # ``tail.pdf(-x)`` is the GPD density of the left tail at ``x``.
        self._tail = scist.genpareto(self.xi, loc=-self.a, scale=self.b)

    def _calc_a(self) -> float:
        """Equation 8 with the sign of the second term flipped: ``a = mu - sig sqrt(W(z))``."""
        return self.mu - self.sig * self.s

    def _calc_sig(self) -> float:
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


class CarBenHybrid:
    """
    Light wrapper that reads the arguments and returns either a
    :class:`CarBenLeft` or a :class:`CarBenRight`.

    Sign convention (kept from 0.0.x; see CHANGELOG):

    * ``xi >= 0`` and ``rtail=True`` (default) -> ``CarBenRight``
    * ``xi >= 0`` and ``rtail=False``          -> ``CarBenLeft``
    * ``xi < 0``                                -> ``CarBenLeft`` with ``xi = -xi``

    Passing a negative ``xi`` together with ``rtail=False`` is an error.
    ``xi`` may be given positionally or by keyword; ``rtail`` may be given as
    the fifth positional argument or by keyword.
    """

    def __new__(cls, *args, **kwargs):
        args = list(args)
        xi_is_arg = len(args) >= 1
        if not xi_is_arg and 'xi' not in kwargs:
            raise TypeError('CarBenHybrid requires `xi`')
        xi = args[0] if xi_is_arg else kwargs['xi']

        if len(args) >= 5:
            rtail = args.pop(4)
        else:
            rtail = kwargs.pop('rtail', True)

        xi = float(xi)
        if xi < 0:
            if not rtail:
                raise ValueError(
                    'If you provide a negative xi parameter, do not provide `rtail`. '
                    'A left-tailed CarBen is computed automatically.'
                )
            rtail = False
            if xi_is_arg:
                args[0] = -xi
            else:
                kwargs['xi'] = -xi

        return CarBenRight(*args, **kwargs) if rtail else CarBenLeft(*args, **kwargs)


class PhatFit(GenericLikelihoodModel):
    """
    Light wrapper for statsmodels GenericLogLikelihood. Used to mirror scipy or 
    statsmodels fit() method.

    Parameters
    -----------
    endog:      iterable of dependet variable
    exog:       iterable of independet variable
    xi_l:       inverse of tail index of left-tailed generalized Pareto tail. Default: None
    xi_r:       inverse of tail index of right-tailed generalized Pareto tail. Default: None
    tail_est:   func; process used to estimate both left and right tail indices. Default: None

    Attributes
    -----------
    left:               left-tailed CarBenHybrid object
    right:              right-tailed CarBenHybrid object
    args:               list of values of 8 parameters of Phat distribution
    params:             name-value pairs of 8 parameters of Phat distribution
    learnable_params:   name-value pairs of 4 given parameters of Phat distribution

    Methods
    --------
        > handle helper and common distribution functions that are agnostic to tail-direction
    """
    def __init__(self, 
        endog:Iterable, 
        exog:Iterable,
        xi_left:float=None, 
        xi_right:float=None, 
        tail_est:Callable=None,
        **kwds):
        
        if xi_right is None and xi_left is None and tail_est is not None:
            self.xi_left, self.xi_right = tail_est(endog)
        else:
            self.xi_left, self.xi_right = xi_left, xi_right
            
        super().__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        """
        Adjust Phat object parameters based on arguments provided at instantiation
        """
        if len(params) == 4:
            phatdist = Phat(*params)
        else:
            phatdist = Phat(params[0], params[1], self.xi_left, self.xi_right)
        return phatdist.nll(self.endog)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        """
        Determines start parameters of fit method based on arguments provided at instantiation

        start_params for Gaussian are set via scipy `norm` distribution.
        """
        if start_params == None:
            if self.xi_left is None and self.xi_right is None:
                self.exog_names.append('xi_l')
                self.exog_names.append('xi_r')
                start_params = np.zeros(4)
                start_params[:2] = scist.norm.fit(self.endog)
                start_params[-2:] = [.2, .2]
            else:
                start_params = np.zeros(2)
                start_params = scist.norm.fit(self.endog)
            
        return super().fit(
            start_params=start_params,
            maxiter=maxiter, maxfun=maxfun,
            **kwds
        )

class Phat:
    """
    Twin-tailed distribution that combines a right-tailed CarBen with a left-tailed CarBen

    Parameters:
        mu:    
            mean of the Gaussian body
        sig:
            standard deviation of the Gaussian body
        xi_l:   
            inverse of tail index of left-tailed generalized Pareto tail
        xi_r:   
            inverse of tail index of right-tailed generalized Pareto tail
        p:
            relative weights of the two CarBen hybrids. Default: [.5, .5]

    Attributes:
        left:               
            left-tailed CarBenHybrid object
        right:              
            right-tailed CarBenHybrid object

    **Details**

    Per Carreau and Bengio (2008).
    
    The properites of the body in both distributions are equivalent; thus, the mixture requires just
    four parameters: two parameters for the Gaussian body and the tail index for each of the two tails. 
    In all, the distribution has 8 parameters:

    + Left tail: xi, a, b
    + Right tail: xi, a, b
    + Center: mu, sig

    As with any mixture model, this mix is the weighted average result of values from the two
    components. Default weights are 0.5 / 0.5.

    Weights could otherwise be estimated via machine learning, threshold analysis (for the tails),
    and gaussian statistics (for the body).
    """
    PARAM_NAMES = ['mu', 'sig', 'xi_l',  'xi_r', 'a_l', 'a_r', 'b_l', 'b_r']

    def __init__(
        self, mu:float, sig:float, xi_l:float, xi_r:float, p:Iterable=None
        ) -> None:
        """
        Left and right tails are instantiated immediately
        """
        self.mu = mu
        self.sig = sig
        self.xi_l = xi_l if xi_l >= 0 else -xi_l
        self.xi_r = xi_r

        self.left = CarBenHybrid(xi_l, mu, sig, rtail=False)
        self.right = CarBenHybrid(xi_r, mu, sig)

        self.p = p if p is not None else np.array([.5,.5])
    
    @property
    def args(self) -> tuple:
        """
        Tuple of the distribution parameters
        """
        return self.mu, self.sig, self.xi_l, self.xi_r, self.left.a, \
            self.right.a, self.left.b, self.right.b

    @property
    def params(self) -> dict:
        """
        Dictionary of the distribution parameters
        """      
        return {name: arg for name, arg in zip(self.PARAM_NAMES, self.args)}

    @property
    def learnable_params(self) -> List[float]:
        """
        List of distribution parameters that can be tuned by machine learning approach
        """
        return [self.params[name] for name in self.PARAM_NAMES[:4]]

    @argsetter('x')
    @dotweight
    def pdf(self, x:XTYPE=None) -> np.ndarray:
        return stacker(x)((self.left.pdf(x), self.right.pdf(x)))
    
    @argsetter('x')
    @dotweight    
    def cdf(self, x:XTYPE=None) -> np.ndarray:
        return stacker(x)((self.left.cdf(x), self.right.cdf(x)))
    
    @argsetter('x')
    def sf(self, x:XTYPE=None) -> np.ndarray:
        return 1 - self.cdf(x)
    
    @dotweight
    def ppf(self, q:XTYPE=None) -> np.ndarray:
        """
        Parameters:
            q: float, iterable of floats 
                Between [0, 1]
        """        
        return stacker(q)((self.left.ppf(q), self.right.ppf(q)))

    def loglike(self, x=None):
        "Loglikelihood"
        return np.log(self.pdf(x))

    @argsetter('x')
    def nll(self, x:XTYPE=None) -> np.ndarray:
        """Negative loglikelihood used in machine learning cost functions"""
        return -np.log(self.pdf(x))

    def rvs(self, 
        size:Union[int, Iterable[Sequence[int]]]=None, 
        seed:int=None) -> XTYPE:
        """
        Generates random draws

        **Process**

        1. Draws randomly from [0,1] uniform distribution
        2. Calls quantile function for each draw

        Mimics `scipy process <https://github.com/scipy/scipy/blob/28e4811d48c99b7c68e41992eda5cff859c1fa2b/scipy/stats/_distn_infrastructure.py#L1032>`_.
        """
        random_state = check_random_state(seed)
        U = random_state.uniform(size=size)
        if size is None:
            U = np.array(U, ndmin=1)
            Y = self.ppf(U)[0]
        else:
            Y = self.ppf(U)
        
        return Y

    @dotweight
    def mean(self) -> float:
        return np.array([self.left.mean(), self.right.mean()])

    @dotweight
    def var(self) -> float:
        return np.array([self.left.var(), self.right.var()])

    def std(self) -> float:
        return np.sqrt(self.var())

    def std_rvs(self, *args, **kwargs) -> XTYPE:
        """
        Generates random variables standardized by dividing by the standard deviation

        Resulting samples should have a standard deviation of 1
        """
        return self.rvs(*args, **kwargs) / self.std()

    @staticmethod
    def fit(values, xi_l:float=None, xi_r:float=None) -> PhatFit:
        """
        Generates standard log-likelihood fit to Phat distribution

        Returns:
            PhatFit
        """
        exog = sm.add_constant(np.zeros_like(values), prepend=True)
        phatfit = PhatFit(values, exog, xi_l, xi_r)

        return phatfit.fit()  

class PhatStack(Phat):
    """
    Weighted model of multiple Phat distributions. 
    
    This is NOT a mixture model. Used only to generate a two dimensional set
    of random variables that are consistent in a single component
    distribution across the second dimension.
    
    Parameters
    -----------
    phats:  iterable or args Phat distributions
    p:      iterable of floats; represents weighting of each Phat distribution
            in the final model. Must sum to 1.

    Attributes
    -----------
    n_comps:      int; number of component phat distributions
    
    Methods
    --------
    rvs
    
    """    
    def __init__(
        self, 
        *phats:Phat,
        p:Iterable=None
        ) -> None:
        """
        """
        if len(phats) == 1:
            phats = phats[0]
        
        if len(phats) != len(p):
            txt = 'Length of probabilities does not match number' 
            txt += f' of distributions: {len(phats)} v {len(p)}'
            raise AssertionError(txt)
        
        self.n_comps = len(phats)
        self.phats = phats
        self.p = arrayarize(p)

    def rvs(self, 
        size:Iterable[int]=None, 
        seed:int=None,
        return_splits:bool=False,
        ) -> XTYPE:
        """
        Generates array of random draws.
        
        ***IMPORTANT***
        Accepts only 2D m x n array. Draws are taken proportionately from
        the component Phat distributions ONLY along the first axis. Along the 
        second axis, draw are taken ONLY from the specific Phat distribution 
        isolated in the first axis.
        
        This ensures that only one distribution regime impacts the time series
        over time.
        
        Where the required iterations, m, does not divide evenly into the
        proportions, the remainders are added to certain distributions

        Params
        -------
        size:           iterable; m x n. Only two dimensions allowed.
        seed:           int; initialize with specific random state for replication
        return_splits:  bool; optional return of the number of m iterations
                        returned for each component distribution
        """
        if len(size) != 2:
            raise AssertionError('Size must be 2D m x n array')

        iters = np.floor(size[0] * self.p).astype(int)
        rmdr = size[0] - iters.sum()
        iters[:rmdr] += np.ones(rmdr, dtype=int)
        
        Y = np.zeros(size)
        st = 0
        for i, end in enumerate(iters):
            Y[st: st+end] = self.phats[i].rvs((end, size[1]), seed=seed)
            st += end
        
        if return_splits:
            return Y, iters
        else:
            return Y
            