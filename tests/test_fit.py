"""Maximum-likelihood estimation recovers the generating parameters."""
import importlib.util

import numpy as np
import pytest

import phat as ph

pytestmark = pytest.mark.skipif(importlib.util.find_spec('statsmodels') is None,
                                reason='Phat.fit needs the fit extra (statsmodels)')


@pytest.fixture(scope='module')
def sample():
    true = dict(mu=0.1, sig=1.0, xi_l=0.2, xi_r=0.3)
    x = ph.Phat(**true).rvs(20_000, random_state=np.random.default_rng(1))
    return true, x


def test_four_parameter_fit_recovers_parameters(sample):
    true, x = sample
    res = ph.Phat.fit(x)
    mu, sig, xi_l, xi_r = res.params
    assert abs(mu - true['mu']) < 0.05
    assert abs(sig - true['sig']) < 0.05
    assert abs(xi_l - true['xi_l']) < 0.05
    assert abs(xi_r - true['xi_r']) < 0.05
    assert np.isfinite(res.llf)
    assert list(res.model.exog_names)[-2:] == ['xi_l', 'xi_r']


def test_two_stage_fit_with_fixed_tails(sample):
    true, x = sample
    res = ph.Phat.fit(x, xi_l=true['xi_l'], xi_r=true['xi_r'])
    assert res.params.shape == (2,)
    assert abs(res.params[0] - true['mu']) < 0.05
    assert abs(res.params[1] - true['sig']) < 0.05


def test_fit_log_likelihood_matches_the_distribution(sample):
    true, x = sample
    res = ph.Phat.fit(x, xi_l=true['xi_l'], xi_r=true['xi_r'])
    dist = ph.Phat(res.params[0], res.params[1], true['xi_l'], true['xi_r'])
    assert np.isclose(res.llf, dist.logpdf(x).sum())


def test_infeasible_parameters_score_inf():
    from phat.fit import PhatFit
    x = np.array([0.1, -0.2, 0.3])
    model = PhatFit(x, np.ones((3, 1)), 0.2, 0.2)
    assert np.isinf(model.nloglikeobs(np.array([0.0, -1.0]))).all()   # sig < 0
    assert np.isfinite(model.nloglikeobs(np.array([0.0, 1.0]))).all()


def test_phatfit_is_reachable_from_old_import_paths():
    from phat.fit import PhatFit
    assert ph.PhatFit is PhatFit
    assert ph.dists.PhatFit is PhatFit
