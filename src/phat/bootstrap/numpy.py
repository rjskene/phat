"""
Numpy-based functions for estimating tail index
via the Hill Double Bootstrap method.

Process is effectively detailed in Voitalov 2019:
https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.1.033034

See Documentation for more.
"""

from typing import Iterable, Union, Tuple
import numpy as np

import numba as nb
from tqdm.auto import trange

@nb.njit
def A_dani(n1, k1):
    return (np.log(k1)/(2*np.log(n1) - np.log(k1)))**(2*(np.log(n1) - np.log(k1))/(np.log(n1)))

@nb.njit
def A_qi(n1, k1):
    return (1 - (2*(np.log(k1) - np.log(n1))/(np.log(k1))))**(np.log(k1)/np.log(n1) - 1)

@nb.njit
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

@nb.njit
def hill_est_for_alpha(k, y):
    return k / (np.cumsum(np.log(y[:-1])) - k*np.log(y[:-1]))

@nb.njit
def hill_est_for_xi(k,y):
    return np.cumsum(np.log(y[:-1]))/k - np.log(y[1:])

@nb.njit
def second_moment(k, y):
    t1 = np.cumsum(np.log(y[:-1])**2) / k 
    t2 = 2*np.cumsum(np.log(y[:-1]))*np.log(y[1:]) / k
    t3 = np.log(y[1:])**2
    return t1 - t2 + t3

@nb.njit
def third_moment(k,y):
    """
    """
    t1 = (1/k)*np.cumsum(np.log(y[:-1])**3)
    t2 = (3*np.log(y[1:])/k)*np.cumsum(np.log(y[:-1])**2)
    t3 = (3*np.log(y[1:])**2/k)*np.cumsum(np.log(y[:-1]))
    t4 = np.log(y[1:])**3
    M3 = t1 - t2 + t3 - t4
    return M3

@nb.njit
def amse(M1, M2):
    return (M2 - 2*M1**2)**2

@nb.njit
def hill_amse(k,y):
    M1 = hill_est_for_xi(k,y)
    M2 = second_moment(k,y)
    return amse(M1,M2)    

@nb.njit
def moments_amse(k, y):
    M1 = hill_est_for_xi(k, y)
    M2 = second_moment(k,y)    
    M3 = third_moment(k, y)
    xi_2 = M1 + 1 - 0.5*((1 - (M1*M1)/M2))**(-1)
    xi_3 = np.sqrt(0.5*M2) + 1 - (2/3)*(1 / (1 - M1*M2/M3))
    return (xi_2 - xi_3)**2

@nb.njit
def finder_loop(y, n, r, style='hill'):
    amses = np.zeros(n-1)
    for i in range(r):
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

def two_tailed_hill_double_bootstrap(
    values:Iterable[float], 
    iters:int=10, 
    return_mean:bool=True,
    pbar_kwargs:dict={},
    ) -> Union[Tuple[Iterable], Tuple[float]]:
    
    left = values[values < values.mean()]
    left = np.sort(-left)[::-1]

    right = values[values > values.mean()]
    right = np.sort(right)[::-1]

    np.seterr(all='ignore')
    shl = np.zeros(iters)
    shr = np.zeros(iters)
    for i in trange(iters, **pbar_kwargs):
        shl[i] = dbl_bs(left, t=.5, r=500, style='hill', A_type='dani')
        shr[i] = dbl_bs(right, t=.5, r=500, style='hill', A_type='dani')
    
    if return_mean:
        return shl.mean(), shr.mean()
    else:
        return shl, shr
