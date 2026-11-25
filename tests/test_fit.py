"""Maximum-likelihood estimation recovers the generating parameters."""
import importlib.util

import numpy as np
import pytest

import phat as ph
from phat.fit import PhatFitResult, fit_phat

HAS_STATSMODELS = importlib.util.find_spec('statsmodels') is not None
TRUE = dict(mu=0.1, sigma=1.0, xi_left=0.2, xi_right=0.3)


@pytest.fixture(scope='module')
def sample():
    return ph.Phat(**TRUE).rvs(20_000, random_state=np.random.default_rng(1))


@pytest.fixture(scope='module')
def full_fit(sample):
    return ph.Phat.fit(sample)


@pytest.fixture(scope='module')
def two_stage_fit(sample):
    return ph.Phat.fit(sample, xi_left=TRUE['xi_left'], xi_right=TRUE['xi_right'])


def test_default_backend_needs_only_numpy_and_scipy(full_fit):
    assert isinstance(full_fit, PhatFitResult)


def test_four_parameter_fit_recovers_parameters(full_fit):
    assert full_fit.names == ('mu', 'sigma', 'xi_left', 'xi_right')
    assert full_fit.converged
    for est, (name, truth) in zip(full_fit.params, TRUE.items()):
        assert abs(est - truth) < 0.05, name
    assert np.isfinite(full_fit.llf)
    assert full_fit.nobs == 20_000


def test_standard_errors_are_finite_and_cover_the_truth(full_fit):
    assert full_fit.bse.shape == (4,)
    assert np.all(np.isfinite(full_fit.bse)) and np.all(full_fit.bse > 0)
    lo, hi = full_fit.conf_int(alpha=0.001).T          # 99.9% intervals
    for name, truth in TRUE.items():
        i = full_fit.names.index(name)
        assert lo[i] < truth < hi[i], name
    assert np.allclose(full_fit.bse, np.sqrt(np.diag(full_fit.cov)))


def test_two_stage_fit_with_fixed_tails(two_stage_fit):
    res = two_stage_fit
    assert res.names == ('mu', 'sigma')
    assert res.params.shape == (2,) and res.bse.shape == (2,)
    assert abs(res.params[0] - TRUE['mu']) < 0.05
    assert abs(res.params[1] - TRUE['sigma']) < 0.05
    assert res.fixed['xi_left'] == TRUE['xi_left'] and res.fixed['xi_right'] == TRUE['xi_right']
    assert np.all(np.isfinite(res.bse))


def test_result_dist_and_log_likelihood_match_the_distribution(sample, two_stage_fit):
    dist = two_stage_fit.dist
    assert isinstance(dist, ph.Phat)
    assert dist.xi_left == TRUE['xi_left'] and dist.xi_right == TRUE['xi_right']
    assert np.isclose(two_stage_fit.llf, dist.logpdf(sample).sum())
    assert np.isclose(two_stage_fit.aic, 2 * 2 - 2 * two_stage_fit.llf)
    assert np.isclose(two_stage_fit.bic, 2 * np.log(len(sample)) - 2 * two_stage_fit.llf)


def test_fixing_one_tail_only_is_rejected(sample):
    with pytest.raises(ValueError):
        ph.Phat.fit(sample, xi_left=0.2)


def test_fit_with_fixed_mixture_weights(sample):
    res = fit_phat(sample, weights=(0.4, 0.6))
    assert res.fixed['weights'] == (0.4, 0.6)
    assert np.allclose(res.dist.weights, (0.4, 0.6))
    assert res.llf < ph.Phat.fit(sample).llf     # the wrong weights fit worse


def test_summary_mentions_every_parameter(full_fit):
    text = full_fit.summary()
    for name in ('mu', 'sigma', 'xi_left', 'xi_right'):
        assert name in text


def test_unknown_backend_is_rejected(sample):
    with pytest.raises(ValueError):
        ph.Phat.fit(sample, backend='mle')


def test_phatfit_is_lazy_in_the_package_namespace():
    import phat.fit as fit_module
    assert 'PhatFit' in dir(fit_module)
    assert 'PhatFit' in dir(ph)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="the statsmodels backend needs the fit extra")
class TestStatsmodelsBackend:
    def test_matches_the_scipy_fit(self, sample, full_fit):
        res = ph.Phat.fit(sample, backend='statsmodels')
        assert list(res.model.exog_names) == ['mu', 'sigma', 'xi_left', 'xi_right']
        assert np.allclose(res.params, full_fit.params, atol=2e-3)
        assert np.allclose(res.bse, full_fit.bse, rtol=0.05)
        assert np.isclose(res.llf, full_fit.llf, atol=0.05)

    def test_infeasible_parameters_score_inf(self):
        from phat.fit import PhatFit
        x = np.array([0.1, -0.2, 0.3])
        model = PhatFit(x, np.ones((3, 1)), 0.2, 0.2)
        assert np.isinf(model.nloglikeobs(np.array([0.0, -1.0]))).all()   # sigma < 0
        assert np.isfinite(model.nloglikeobs(np.array([0.0, 1.0]))).all()

    def test_phatfit_is_reachable_from_old_import_paths(self):
        from phat.fit import PhatFit
        assert ph.PhatFit is PhatFit
        assert ph.dists.PhatFit is PhatFit

    def test_statsmodels_backend_rejects_weights(self, sample):
        with pytest.raises(ValueError):
            ph.Phat.fit(sample, weights=(0.4, 0.6), backend='statsmodels')
