"""Hill double bootstrap: regression on a committed text fixture, sanity and reproducibility."""
import importlib.util
import pathlib

import numpy as np
import pytest

from phat.bootstrap import seed_bootstrap, two_tailed_hill_double_bootstrap

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'student_t3_n1500.csv'
HAS_NUMBA = importlib.util.find_spec('numba') is not None


@pytest.fixture(scope='module')
def t3_sample():
    return np.loadtxt(FIXTURE)


def test_fixture_is_as_documented(t3_sample):
    assert t3_sample.shape == (1500,)
    regenerated = np.random.default_rng(20261125).standard_t(3, 1500)
    assert np.allclose(t3_sample, regenerated, atol=1e-9)


def test_estimates_in_a_sane_range(t3_sample):
    # true xi = 1/3 on both tails; the Hill estimator is biased upward for Student t at this n
    xi_l, xi_r = two_tailed_hill_double_bootstrap(t3_sample, iters=2, seed=1, pbar_kwargs={'disable': True})
    assert 0.2 < xi_l < 0.6
    assert 0.2 < xi_r < 0.6


def test_seed_makes_the_estimate_reproducible(t3_sample):
    kw = dict(iters=2, pbar_kwargs={'disable': True})
    a = two_tailed_hill_double_bootstrap(t3_sample, seed=7, **kw)
    b = two_tailed_hill_double_bootstrap(t3_sample, seed=7, **kw)
    c = two_tailed_hill_double_bootstrap(t3_sample, seed=8, **kw)
    assert a == b
    assert a != c


def test_return_every_run(t3_sample):
    shl, shr = two_tailed_hill_double_bootstrap(
        t3_sample, iters=3, return_mean=False, seed=1, pbar_kwargs={'disable': True})
    assert shl.shape == shr.shape == (3,)
    assert np.isfinite(shl).all() and np.isfinite(shr).all()


@pytest.mark.skipif(not HAS_NUMBA, reason='regression values recorded on the Numba path')
def test_regression_numba_path(t3_sample):
    # Recorded 2026-11-25 with numba 0.68 / numpy 2.5 on arm64. Both RNG paths are seeded
    # by seed_bootstrap; the Numba stream is the one pinned here.
    xi_l, xi_r = two_tailed_hill_double_bootstrap(t3_sample, iters=2, seed=1, pbar_kwargs={'disable': True})
    assert np.isclose(xi_l, 0.45221686311874376, rtol=1e-9)
    assert np.isclose(xi_r, 0.4598391828303687, rtol=1e-9)


def test_accepts_lists_and_leaves_global_errstate_alone(t3_sample):
    before = np.geterr()
    seed_bootstrap(3)
    two_tailed_hill_double_bootstrap(list(t3_sample[:600]), iters=1, pbar_kwargs={'disable': True})
    assert np.geterr() == before
