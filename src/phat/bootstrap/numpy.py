"""
NumPy-based functions for estimating the tail index via the Hill double
bootstrap (Danielsson, de Haan, Peng and de Vries, 2001; Qi, 2008).

Attribution
-----------
The functions in this module are adapted from ``tail-estimation`` by Ivan
Voitalov (https://github.com/ivanvoitalov/tail-estimation), released under
the MIT License, which accompanies the paper

    Voitalov, van der Hoorn, van der Hofstad and Krioukov, "Scale-free
    networks well done", Physical Review Research 1, 033034 (2019),
    https://doi.org/10.1103/PhysRevResearch.1.033034

The two-tailed wrapper :func:`two_tailed_hill_double_bootstrap` splits the
sample at its mean and applies the procedure to each side.

Numba, when installed, compiles the inner loops; tqdm, when installed, shows a
progress bar. Neither is required.
"""

from collections.abc import Iterable
from typing import Union

import numpy as np

from phat._compat import renamed_kwargs

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

try:
    from tqdm.auto import trange
except ImportError:  # pragma: no cover
    def trange(n, **kwargs):
        return range(n)

@_njit
def A_dani(n1, k1):
    return (np.log(k1)/(2*np.log(n1) - np.log(k1)))**(2*(np.log(n1) - np.log(k1))/(np.log(n1)))

@_njit
def A_qi(n1, k1):
    return (1 - (2*(np.log(k1) - np.log(n1))/(np.log(k1))))**(np.log(k1)/np.log(n1) - 1)

@_njit
def moments_dbs_prefactor(xi_n, n1, k1):
    """
    Function to calculate pre-factor used in moments
    double-bootstrap procedure.

    Params
    -------
    xi_n: moments tail index estimate corresponding to
            sqrt(n)-th order statistic.
    n1:   size of the 1st bootstrap in double-bootstrap
            procedure.
    k1:   estimated optimal order statistic based on the 1st
            bootstrap sample.

    Returns
    --------
    prefactor: constant used in estimation of the optimal
                stopping order statistic for moments estimator.
    """
    def V_sq(xi_n):
        if xi_n >= 0:
            V = 1. + (xi_n)**2
            return V
        else:
            a = (1.-xi_n)**2
            b = (1-2*xi_n)*(6*((xi_n)**2)-xi_n+1)
            c = (1.-3*xi_n)*(1-4*xi_n)
            V = a*b/c
            return V

    def V_bar_sq(xi_n):
        if xi_n >= 0:
            V = 0.25*(1+(xi_n)**2)
            return V
        else:
            a = 0.25*((1-xi_n)**2)
            b = 1-8*xi_n+48*(xi_n**2)-154*(xi_n**3)
            c = 263*(xi_n**4)-222*(xi_n**5)+72*(xi_n**6)
            d = (1.-2*xi_n)*(1-3*xi_n)*(1-4*xi_n)
            e = (1.-5*xi_n)*(1-6*xi_n)
            V = a*(b+c)/(d*e)
            return V

    def b(xi_n, rho):
        if xi_n < rho:
            a1 = (1.-xi_n)*(1-2*xi_n)
            a2 = (1.-rho-xi_n)*(1.-rho-2*xi_n)
            return a1/a2
        elif xi_n >= rho and xi_n < 0:
            return 1./(1-xi_n)
        else:
            b = (xi_n/(rho*(1.-rho))) + (1./((1-rho)**2))
            return b

    def b_bar(xi_n, rho):
        if xi_n < rho:
            a1 = 0.5*(-rho*(1-xi_n)**2)
            a2 = (1.-xi_n-rho)*(1-2*xi_n-rho)*(1-3*xi_n-rho)
            return a1/a2
        elif xi_n >= rho and xi_n < 0:
            a1 = 1-2*xi_n-np.sqrt((1-xi_n)*(1-2.*xi_n))
            a2 = (1.-xi_n)*(1-2*xi_n)
            return a1/a2
        else:
            b = (-1.)*((rho + xi_n*(1-rho))/(2*(1-rho)**3))
            return b

    rho = np.log(k1)/(2*np.log(k1) - 2.*np.log(n1))
    a = (V_sq(xi_n)) * (b_bar(xi_n, rho)**2)
    b = V_bar_sq(xi_n) * (b(xi_n, rho)**2)
    prefactor = (a/b)**(1./(1. - 2*rho))
    return prefactor

@_njit
def hill_est_for_alpha(k, y):
    return k / (np.cumsum(np.log(y[:-1])) - k*np.log(y[:-1]))

@_njit
def hill_est_for_xi(k,y):
    return np.cumsum(np.log(y[:-1]))/k - np.log(y[1:])

@_njit
def second_moment(k, y):
    t1 = np.cumsum(np.log(y[:-1])**2) / k
    t2 = 2*np.cumsum(np.log(y[:-1]))*np.log(y[1:]) / k
    t3 = np.log(y[1:])**2
    return t1 - t2 + t3

@_njit
def third_moment(k,y):
    """
    """
    t1 = (1/k)*np.cumsum(np.log(y[:-1])**3)
    t2 = (3*np.log(y[1:])/k)*np.cumsum(np.log(y[:-1])**2)
    t3 = (3*np.log(y[1:])**2/k)*np.cumsum(np.log(y[:-1]))
    t4 = np.log(y[1:])**3
    M3 = t1 - t2 + t3 - t4
    return M3

@_njit
def amse(M1, M2):
    return (M2 - 2*M1**2)**2

@_njit
def hill_amse(k,y):
    M1 = hill_est_for_xi(k,y)
    M2 = second_moment(k,y)
    return amse(M1,M2)

@_njit
def moments_amse(k, y):
    M1 = hill_est_for_xi(k, y)
    M2 = second_moment(k,y)
    M3 = third_moment(k, y)
    xi_2 = M1 + 1 - 0.5*(1 - (M1*M1)/M2)**(-1)
    xi_3 = np.sqrt(0.5*M2) + 1 - (2/3)*(1 / (1 - M1*M2/M3))
    return (xi_2 - xi_3)**2

@_njit
def finder_loop(y, n, r, style='hill'):
    amses = np.zeros(n-1)
    for _ in range(r):
        sample = np.random.choice(y, n, replace=False)
        sample = np.sort(sample)[::-1]
        k = np.arange(1,n)

        if style == 'hill':
            amses += hill_amse(k,sample)
        elif style =='moments':
            amses += moments_amse(k,sample)

    return amses

def k_finder(y, n, r, kmin, style='hill'):
    kmax = (np.abs(np.linspace(1./n, 1.0, n) - 1)).argmin()
    amses = finder_loop(y, n, r, style)
    amse_for_k = amses / n

    k = np.nanargmin(amse_for_k[kmin:kmax]) + 1 + kmin
    return k

def dbl_bs(y, t=.5, r=500, style='hill', A_type='qi'):
    n = y.size
    n1 = int(np.sqrt(t)*n)
    n2 = int(t*n)
    k = np.arange(1, y.size)
    xi = hill_est_for_xi(k,y)
    xi_n = xi[int(np.floor(n**0.5))-1]

    kmin1, kmin2 = 1,1
    while True:
        k1 = k_finder(y, n1, r, kmin1, style=style)
        k2 = k_finder(y, n2, r, kmin2, style=style)

        if k2 > k1:
            kmin1 += int(0.005*n)
            kmin2 += int(0.005*n)
        else:
            break

    A = A_qi if A_type == 'qi' else A_dani

    if style == 'hill':
        prefactor = A(n1,k1)
    elif style =='moments':
        prefactor = moments_dbs_prefactor(xi_n, n1, k1)
    else:
        raise ValueError('`style` not supported')

    k_star = prefactor*k1**2 / k2
    k_star = np.round(k_star).astype(int)

    if k_star >= n:
        raise ValueError(f'Estimated threshold larger than size of sample data: k {k_star} v. n {n}')
    else:
        k_star = 2 if k_star == 0 else k_star

    return xi[k_star]

@_njit
def _seed_jit(seed):
    np.random.seed(seed)


def seed_bootstrap(seed:int) -> None:
    """
    Seed the random number generator used by the bootstrap resampling.

    The resampling runs inside Numba-compiled code when Numba is installed,
    which keeps its own random state; this seeds that state (and NumPy's global
    state, which the pure-Python fallback uses). Results for a given seed differ
    between the Numba and pure-Python paths.
    """
    np.random.seed(seed)
    _seed_jit(seed)


@renamed_kwargs(iters='n_runs')
def two_tailed_hill_double_bootstrap(
    values:Iterable[float],
    n_runs:int=10,
    return_mean:bool=True,
    pbar_kwargs:dict=None,
    seed:int=None,
    ) -> Union[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    """
    Hill double-bootstrap estimate of the tail shape ``xi`` on each side of the
    sample mean.

    Parameters
    ----------
    values:         1-d sample
    n_runs:          number of independent double-bootstrap runs to average
    return_mean:    return the mean over runs (``True``) or every run
    pbar_kwargs:    keyword arguments for the tqdm progress bar
    seed:           seed for the bootstrap resampling (see :func:`seed_bootstrap`)

    Returns
    -------
    ``(xi_left, xi_right)`` as floats, or as arrays of length ``n_runs``
    """
    values = np.asarray(values, dtype=float)
    pbar_kwargs = {} if pbar_kwargs is None else pbar_kwargs
    if seed is not None:
        seed_bootstrap(seed)

    left = values[values < values.mean()]
    left = np.sort(-left)[::-1]

    right = values[values > values.mean()]
    right = np.sort(right)[::-1]

    shl = np.zeros(n_runs)
    shr = np.zeros(n_runs)
    with np.errstate(all='ignore'):
        for i in trange(n_runs, **pbar_kwargs):
            shl[i] = dbl_bs(left, t=.5, r=500, style='hill', A_type='dani')
            shr[i] = dbl_bs(right, t=.5, r=500, style='hill', A_type='dani')

    if return_mean:
        return float(shl.mean()), float(shr.mean())
    else:
        return shl, shr
