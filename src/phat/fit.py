"""
Maximum-likelihood estimation of the Phat distribution with ``statsmodels``.

This module imports ``statsmodels`` at load time and is therefore imported
lazily by :meth:`phat.dists.Phat.fit`. Install the extra with
``pip install "phat-tails[fit]"``.
"""
from typing import Callable, Iterable
import numpy as np
import scipy.stats as scist

from statsmodels.base.model import GenericLikelihoodModel as _GenericLikelihoodModel

from phat.dists import Phat

__all__ = ['PhatFit']




class PhatFit(_GenericLikelihoodModel):
    """
    Light wrapper around ``statsmodels`` ``GenericLikelihoodModel`` so that
    :meth:`Phat.fit` mirrors the ``scipy`` / ``statsmodels`` ``fit()`` interface.

    Parameters
    -----------
    endog:      iterable of observations
    exog:       iterable of regressors (a constant column; the model has no covariates)
    xi_left:    shape of the left tail. ``None`` to estimate it.
    xi_right:   shape of the right tail. ``None`` to estimate it.
    tail_est:   callable returning ``(xi_left, xi_right)`` from ``endog``; used
                when both shapes are ``None``.
    """

    def __init__(self,
        endog: Iterable,
        exog: Iterable,
        xi_left: float = None,
        xi_right: float = None,
        tail_est: Callable = None,
        **kwds):
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
        """
        Start values: Gaussian ``norm.fit`` for ``(mu, sig)``; ``0.2`` for a free tail shape.
        """
        if start_params is None:
            if self.xi_left is None and self.xi_right is None:
                if 'xi_l' not in self.exog_names:
                    self.exog_names.append('xi_l')
                    self.exog_names.append('xi_r')
                start_params = np.zeros(4)
                start_params[:2] = scist.norm.fit(self.endog)
                start_params[-2:] = [.2, .2]
            else:
                start_params = np.asarray(scist.norm.fit(self.endog))

        return super().fit(
            start_params=start_params,
            maxiter=maxiter, maxfun=maxfun,
            **kwds
        )
