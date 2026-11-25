"""
Independent re-implementation of the Phat distribution for testing.

Everything here is built directly from SciPy's ``norm`` and ``genpareto`` plus
quadrature and scalar root-finding, following the model statement in
``docs``: right component = Carreau-Bengio hybrid Pareto, left component = its
reflection about ``mu``, Phat = p_l * left + p_r * right. It deliberately
shares no code with ``phat.dists``.
"""
from __future__ import annotations

import numpy as np
import scipy.special as sp
import scipy.stats as st
from scipy.integrate import quad
from scipy.optimize import brentq


def _component(xi, mu, sig):
    w = sp.lambertw((1 + xi) ** 2 / (2 * np.pi)).real
    s = np.sqrt(w)
    return dict(xi=xi, mu=mu, sig=sig, s=s, b=sig * (1 + xi) / s, gam=1 + st.norm.cdf(s))


def pdf_right(x, c):
    a = c["mu"] + c["sig"] * c["s"]
    return np.where(
        x > a,
        st.genpareto.pdf(x, c["xi"], loc=a, scale=c["b"]),
        st.norm.pdf(x, c["mu"], c["sig"]),
    ) / c["gam"]


def cdf_right(x, c):
    a = c["mu"] + c["sig"] * c["s"]
    return np.where(
        x > a,
        st.norm.cdf(a, c["mu"], c["sig"]) + st.genpareto.cdf(x, c["xi"], loc=a, scale=c["b"]),
        st.norm.cdf(x, c["mu"], c["sig"]),
    ) / c["gam"]


def pdf_left(x, c):
    return pdf_right(2 * c["mu"] - x, c)


def cdf_left(x, c):
    return 1 - cdf_right(2 * c["mu"] - x, c)


class OraclePhat:
    """Reference Phat(mu, sig, xi_l, xi_r, p) with quadrature moments and root quantiles."""

    def __init__(self, mu, sig, xi_l, xi_r, p=(0.5, 0.5)):
        self.mu, self.sig, self.xi_l, self.xi_r = mu, sig, xi_l, xi_r
        self.p = np.asarray(p, float)
        self.L = _component(xi_l, mu, sig)
        self.R = _component(xi_r, mu, sig)
        self.a_l = mu - sig * self.L["s"]
        self.a_r = mu + sig * self.R["s"]

    def pdf(self, x):
        x = np.asarray(x, float)
        return self.p[0] * pdf_left(x, self.L) + self.p[1] * pdf_right(x, self.R)

    def cdf(self, x):
        x = np.asarray(x, float)
        return self.p[0] * cdf_left(x, self.L) + self.p[1] * cdf_right(x, self.R)

    def sf(self, x):
        return 1 - self.cdf(x)

    def _integrate(self, f, lo=-np.inf, hi=np.inf):
        pts = sorted({lo, hi, self.a_l, self.a_r, self.mu} | set())
        pts = [p for p in pts if lo <= p <= hi]
        if pts[0] != lo:
            pts.insert(0, lo)
        if pts[-1] != hi:
            pts.append(hi)
        return sum(quad(f, u, v, limit=400)[0] for u, v in zip(pts[:-1], pts[1:]))

    def moment(self, k, lo=-np.inf, hi=np.inf):
        return self._integrate(lambda x: x**k * float(self.pdf(x)), lo, hi)

    def mean(self):
        return self.moment(1)

    def var(self):
        m1 = self.moment(1)
        return self.moment(2) - m1**2

    def ppf(self, q):
        q = np.atleast_1d(np.asarray(q, float))
        out = np.empty_like(q)
        for i, qi in enumerate(q):
            out[i] = brentq(lambda x: float(self.cdf(x)) - qi, -1e9, 1e9, xtol=1e-13, rtol=1e-13, maxiter=500)
        return out

    def es_lower(self, q):
        """E[X | X <= ppf(q)]"""
        xq = float(self.ppf(q)[0])
        return self.moment(1, -np.inf, xq) / q

    def es_upper(self, q):
        """E[X | X > isf(q)] = E[X | X > ppf(1 - q)]"""
        xq = brentq(lambda x: float(self.sf(x)) - q, -1e9, 1e9, xtol=1e-13, rtol=1e-13, maxiter=500)
        return self.moment(1, xq, np.inf) / q
