"""
Simulation study behind the "Testing" section of ``dblbs.ipynb``.

Regenerates the two result files read by the notebook:

    python dbs_tests.py 1    # dbs_test_on_r_and_t.csv                 (about 20 minutes)
    python dbs_tests.py 2    # dbs_test_on_sample_size_and_tails.csv   (about 90 minutes)

Every random draw is seeded, so the files are reproducible on a given platform
with Numba installed (the pure-Python fallback uses a different random stream).
"""
import itertools as it
import sys

import numpy as np
import pandas as pd

import phat as ph
from phat.bootstrap.numpy import dbl_bs, seed_bootstrap

REPS = 4


def estimate(y, **kwargs):
    """Hill double-bootstrap estimate, ``nan`` when the procedure finds no threshold."""
    try:
        with np.errstate(invalid='ignore', divide='ignore'):
            return float(dbl_bs(y, style='hill', A_type='qi', **kwargs))
    except ValueError:
        return np.nan


def descending(values):
    return np.sort(values)[::-1]


def study_r_and_t():
    """One sample of 100,000 from Phat(.25, 1.4, .35, .22); vary ``t`` and ``r``."""
    seed_bootstrap(1)
    data = ph.Phat(.25, 1.4, .35, .22).rvs(size=100000, random_state=1)   # the notebook's sample
    y_right = descending(data[data >= data.mean()])
    y_left = descending(-data[data < data.mean()])
    rows = []
    for t, r in it.product(np.linspace(.1, .8, 5), [100, 250, 500, 1000, 2500]):
        for _ in range(REPS):
            rows.append({
                't': t, 'r': r,
                'xi_left_hat': estimate(y_left, t=t, r=r),
                'xi_right_hat': estimate(y_right, t=t, r=r),
            })
        print(f't={t:.3f} r={r}', flush=True)
    pd.DataFrame(rows).to_csv('dbs_test_on_r_and_t.csv', index=False)


def study_size_and_tails():
    """Vary the sample size and the two tail shapes; ``t = .5``, ``r = 500``."""
    seed_bootstrap(2)
    rng = np.random.default_rng(2)
    ns = np.logspace(3.75, 5.35, 5).astype(int)
    tails = [.1, .2, .35, .5, 1, 2.5, 5]
    rows = []
    for n, (xi_left, xi_right) in it.product(ns, it.product(tails, tails)):
        data = ph.Phat(.25, 1.4, xi_left, xi_right).rvs(size=int(n), random_state=rng)
        y_left = descending(-data[data < 0])
        y_right = descending(data[data > 0])
        for _ in range(REPS):
            rows.append({
                't': .5, 'r': 500, 'n': int(n),
                'xi_lactual': xi_left, 'xi_ractual': xi_right,
                'xi_lest': estimate(y_left, t=.5, r=500),
                'xi_rest': estimate(y_right, t=.5, r=500),
            })
        print(f'n={n} xi=({xi_left}, {xi_right})', flush=True)
    pd.DataFrame(rows).to_csv('dbs_test_on_sample_size_and_tails.csv', index=False)


if __name__ == '__main__':
    {'1': study_r_and_t, '2': study_size_and_tails}[sys.argv[1]]()
