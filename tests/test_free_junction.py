"""
Phat with a free junction.

In 1.0 the junction of each hybrid Pareto component is implied by its tail
shape: the Gaussian and the generalized Pareto piece carry the same multiplier
``1 / gamma``. With a junction distance ``s`` given, the Pareto piece carries its
own multiplier ``lambda``; continuity of the density and of its slope give

    b = sigma (1 + xi) / s,   lambda = (1 + xi) phi(s) / s,   gamma = Phi(s) + lambda

and ``lambda = 1`` is the implied junction. The one-tailed density with unequal
weights is that of Debbabi, Kratz and El Asmi (2015); the relations used below
as a cross-check are system (E) of Debbabi, Kratz, El Asmi and Mboup (2015,
GRETSI).
"""
import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st
from scipy.integrate import quad

import phat as ph
from phat.fit import fit_phat
from tests.oracle import OraclePhat

# (mu, sigma, xi_left, xi_right, s_left, s_right)
FREE_CASES = [
    (0.0, 1.0, 0.2, 0.25, 1.0, 1.0),
    (0.1, 0.2, 0.2, 0.25, 1.4, 0.8),
    (-1.5, 0.7, 0.0, 0.45, 0.6, 2.0),
    (0.3, 3.0, 0.6, 0.05, 2.5, 1.2),
    (0.3, 1.5, -0.3, 0.2, 0.9, 1.1),
    (2.0, 2.0, 0.3, 0.3, 0.3, 0.3),          # closer in than the implied junction
]


def implied_s(xi):
    return float(np.sqrt(sp.lambertw((1 + xi) ** 2 / (2 * np.pi)).real))


@pytest.fixture(params=FREE_CASES, ids=lambda c: f"Phat{c}")
def pair(request):
    mu, sigma, xl, xr, sl, sr = request.param
    dist = ph.Phat(mu, sigma, xl, xr, s_left=sl, s_right=sr)
    return dist, OraclePhat(mu, sigma, xl, xr, s_l=sl, s_r=sr)


# ---- components -------------------------------------------------------------
class TestComponent:
    def test_implied_junction_is_lambda_one(self):
        for cls in (ph.CarBenRight, ph.CarBenLeft):
            c = cls(0.25, 1.0, 2.0)
            assert c.lam == 1.0
            assert c.junction == 'implied'
            assert c.s == pytest.approx(implied_s(0.25), rel=1e-14)
            assert c.gamma == 1.0 + st.norm.cdf(c.s)

    def test_free_junction_relations(self):
        xi, mu, sigma, s = 0.3, 0.5, 2.0, 1.25
        c = ph.CarBenRight(xi, mu, sigma, s=s)
        assert c.junction == 'free'
        assert c.s == s
        assert c.a == pytest.approx(mu + s * sigma)
        assert c.b == pytest.approx(sigma * (1 + xi) / s)
        assert c.lam == pytest.approx((1 + xi) * st.norm.pdf(s) / s)
        assert c.gamma == pytest.approx(st.norm.cdf(s) + c.lam)
        assert c.tail_mass == pytest.approx(c.lam / c.gamma)
        assert c.body_mass == pytest.approx(st.norm.cdf(s) / c.gamma)
        assert c.sf(c.a) == pytest.approx(c.tail_mass)
        assert c.qjunc == pytest.approx(1 - c.tail_mass)

    def test_matches_the_published_three_piece_relations(self):
        """System (E), Debbabi et al. (2015): beta = (1 + xi) sigma^2 / (u - mu), weight ratio = beta f(u)."""
        xi, mu, sigma, s = 0.5, 0.0, 1.0, 1.5
        c = ph.CarBenRight(xi, mu, sigma, s=s)
        u = mu + s * sigma
        beta = (1 + xi) * sigma ** 2 / (u - mu)
        assert c.b == pytest.approx(beta)
        assert c.lam == pytest.approx(beta * st.norm.pdf(u, mu, sigma))

    def test_implied_distance_given_explicitly_reproduces_the_implied_junction(self):
        x = np.linspace(-8, 30, 400)
        for cls in (ph.CarBenRight, ph.CarBenLeft):
            ref = cls(0.25, 1.0, 2.0)
            c = cls(0.25, 1.0, 2.0, s=implied_s(0.25))
            assert c.lam == pytest.approx(1.0, abs=1e-13)
            np.testing.assert_allclose(c.pdf(x), ref.pdf(x), rtol=1e-12)
            np.testing.assert_allclose(c.cdf(x), ref.cdf(x), rtol=1e-12)

    def test_three_ways_to_give_a_free_junction(self):
        xi, mu, sigma, s = 0.2, 1.0, 2.0, 1.5
        for cls, sign in ((ph.CarBenRight, 1), (ph.CarBenLeft, -1)):
            a = mu + sign * s * sigma
            by_s = cls(xi, mu, sigma, s=s)
            by_a = cls(xi, mu, sigma, a)
            by_a_s = cls(xi, mu, a=a, s=s)
            for other in (by_a, by_a_s):
                assert other.junction == 'free'
                for name in ('sigma', 'a', 'b', 's', 'lam', 'gamma'):
                    assert getattr(other, name) == pytest.approx(getattr(by_s, name)), name

    def test_factory_passes_the_junction_on(self):
        c = ph.CarBenHybrid(0.2, 0.0, 1.0, s=1.5, tail='left')
        assert isinstance(c, ph.CarBenLeft) and c.s == 1.5 and c.a == pytest.approx(-1.5)

    def test_invalid_junctions_are_rejected(self):
        with pytest.raises(ValueError):
            ph.CarBenRight(0.2, 0.0, 1.0, 1.5, s=1.5)        # sigma, a and s together
        with pytest.raises(ValueError):
            ph.CarBenRight(0.2, 0.0, 1.0, s=0.0)
        with pytest.raises(ValueError):
            ph.CarBenRight(0.2, 0.0, 1.0, s=-1.0)
        with pytest.raises(ValueError):
            ph.CarBenRight(0.2, 0.0, 1.0, a=-0.5)            # junction on the wrong side of mu
        with pytest.raises(ValueError):
            ph.CarBenLeft(0.2, 0.0, 1.0, a=0.5)

    def test_left_is_the_mirror_image(self):
        r = ph.CarBenRight(0.3, 0.5, 2.0, s=1.25)
        left = ph.CarBenLeft(0.3, 0.5, 2.0, s=1.25)
        x = np.linspace(-30, 30, 301)
        np.testing.assert_allclose(left.pdf(x), r.pdf(2 * 0.5 - x), rtol=1e-13)
        np.testing.assert_allclose(left.cdf(x), r.sf(2 * 0.5 - x), rtol=1e-13)
        assert left.a == pytest.approx(0.5 - 1.25 * 2.0)
        assert left.mean() == pytest.approx(2 * 0.5 - r.mean())

    @pytest.mark.parametrize('s', [0.3, 1.0, 2.5])
    def test_component_functions_are_consistent(self, s):
        c = ph.CarBenRight(0.3, 0.5, 2.0, s=s)
        total = quad(c.pdf, -np.inf, c.a)[0] + quad(c.pdf, c.a, np.inf)[0]
        assert total == pytest.approx(1.0, abs=1e-8)
        q = np.array([1e-6, 0.01, 0.3, c.qjunc, 0.9, 0.999, 1 - 1e-9])
        np.testing.assert_allclose(c.cdf(c.ppf(q)), q, rtol=1e-9)
        np.testing.assert_allclose(c.sf(c.isf(q)), q, rtol=1e-9)
        x = np.array([-5.0, 0.0, c.a, c.a + 1, c.a + 50])
        np.testing.assert_allclose(c.cdf(x) + c.sf(x), 1.0, rtol=1e-12)
        np.testing.assert_allclose(np.exp(c.logpdf(x)), c.pdf(x), rtol=1e-12)
        np.testing.assert_allclose(np.exp(c.logcdf(x)), c.cdf(x), rtol=1e-12)
        np.testing.assert_allclose(np.exp(c.logsf(x)), c.sf(x), rtol=1e-12)
        m1 = quad(lambda v: v * c.pdf(v), -np.inf, c.a)[0] + quad(lambda v: v * c.pdf(v), c.a, np.inf)[0]
        second = lambda v: v * v * c.pdf(v)  # noqa: E731
        m2 = quad(second, -np.inf, c.a)[0] + quad(second, c.a, np.inf)[0]
        assert c.mean() == pytest.approx(m1, rel=1e-7)
        assert c.var() == pytest.approx(m2 - m1 ** 2, rel=1e-6)
        for t in (c.a - 1.0, c.a + 2.0):
            lower = quad(lambda v: v * c.pdf(v), -np.inf, min(t, c.a))[0]
            lower += quad(lambda v: v * c.pdf(v), c.a, t)[0] if t > c.a else 0.0
            assert c.lower_partial_mean(t) == pytest.approx(lower, rel=1e-7, abs=1e-9)
            assert c.upper_partial_mean(t) == pytest.approx(m1 - lower, rel=1e-7, abs=1e-9)

    def test_far_junction_tends_to_the_gaussian(self):
        c = ph.CarBenRight(0.3, 0.0, 1.0, s=6.0)
        x = np.linspace(-4, 4, 81)
        np.testing.assert_allclose(c.pdf(x), st.norm.pdf(x), rtol=1e-7)
        assert c.tail_mass < 1e-8


# ---- the mixture ------------------------------------------------------------
def test_default_is_the_implied_junction_and_is_unchanged():
    dist = ph.Phat(0.1, 2.0, 0.2, 0.3)
    assert dist.junction == ('implied', 'implied')
    assert dist.lam_left == 1.0 and dist.lam_right == 1.0
    assert dist.s_left == pytest.approx(implied_s(0.2)) and dist.s_right == pytest.approx(implied_s(0.3))
    same = ph.Phat(0.1, 2.0, 0.2, 0.3, s_left=implied_s(0.2), s_right=implied_s(0.3))
    x = np.linspace(-60, 60, 601)
    np.testing.assert_allclose(same.pdf(x), dist.pdf(x), rtol=1e-12)
    np.testing.assert_allclose(same.cdf(x), dist.cdf(x), rtol=1e-12)
    assert same.mean() == pytest.approx(dist.mean(), rel=1e-12)


def test_attributes_of_a_free_junction():
    dist = ph.Phat(0.1, 2.0, 0.2, 0.3, s_left=1.2, s_right=0.9)
    assert dist.junction == ('free', 'free')
    assert (dist.s_left, dist.s_right) == (1.2, 0.9)
    assert dist.junctions == pytest.approx((0.1 - 1.2 * 2.0, 0.1 + 0.9 * 2.0))
    assert dist.lam_left == pytest.approx(1.2 * st.norm.pdf(1.2) / 1.2)
    assert dist.lam_right == pytest.approx(1.3 * st.norm.pdf(0.9) / 0.9)
    assert dist.params['a_left'] == pytest.approx(-2.3)
    assert dist.params['b_right'] == pytest.approx(2.0 * 1.3 / 0.9)


def test_one_side_free_only():
    dist = ph.Phat(0.0, 1.0, 0.2, 0.3, s_left=1.5)
    assert dist.junction == ('free', 'implied')
    assert dist.lam_right == 1.0 and dist.s_right == pytest.approx(implied_s(0.3))
    oracle = OraclePhat(0.0, 1.0, 0.2, 0.3, s_l=1.5)
    x = np.linspace(-20, 20, 201)
    np.testing.assert_allclose(dist.pdf(x), oracle.pdf(x), rtol=1e-11)


def test_invalid_junction_distance_is_rejected():
    with pytest.raises(ValueError):
        ph.Phat(0.0, 1.0, 0.2, 0.3, s_left=0.0)
    with pytest.raises(ValueError):
        ph.Phat(0.0, 1.0, 0.2, 0.3, s_right=-1.0)


def test_density_integrates_to_one(pair):
    dist, oracle = pair
    assert oracle._integrate(lambda x: float(dist.pdf(x))) == pytest.approx(1.0, abs=1e-7)


def test_pdf_cdf_match_oracle(pair):
    dist, oracle = pair
    x = np.concatenate([np.linspace(-60, 60, 1201) * dist.sigma + dist.mu, list(dist.junctions)])
    np.testing.assert_allclose(dist.pdf(x), oracle.pdf(x), rtol=1e-10, atol=1e-300)
    np.testing.assert_allclose(dist.cdf(x), oracle.cdf(x), rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(dist.cdf(x) + dist.sf(x), 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.exp(dist.logpdf(x)), dist.pdf(x), rtol=1e-11, atol=1e-300)


def test_ppf_inverts_cdf_and_matches_oracle(pair):
    dist, oracle = pair
    q = np.array([1e-5, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 1 - 1e-5])
    x = dist.ppf(q)
    np.testing.assert_allclose(dist.cdf(x), q, rtol=1e-9)
    np.testing.assert_allclose(x, oracle.ppf(q), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(dist.sf(dist.isf(q)), q, rtol=1e-9)


def test_density_and_derivative_continuous_at_junctions(pair):
    dist, _ = pair
    h = 1e-6 * dist.sigma
    for a in dist.junctions:
        assert dist.pdf(a - h) == pytest.approx(dist.pdf(a + h), rel=1e-5)
        d_left = (dist.pdf(a) - dist.pdf(a - h)) / h
        d_right = (dist.pdf(a + h) - dist.pdf(a)) / h
        assert d_left == pytest.approx(d_right, rel=1e-4, abs=1e-8)


def test_moments_match_quadrature(pair):
    dist, oracle = pair
    if max(dist.xi_left, dist.xi_right) < 1:
        assert dist.mean() == pytest.approx(oracle.mean(), rel=1e-6, abs=1e-8)
    if max(dist.xi_left, dist.xi_right) < 0.5:
        assert dist.var() == pytest.approx(oracle.var(), rel=1e-5)
    else:
        assert dist.var() == np.inf


def test_expected_shortfall_matches_quadrature(pair):
    dist, oracle = pair
    for q in (0.01, 0.05, 0.4):
        if dist.xi_left < 1:
            assert dist.expected_shortfall(q) == pytest.approx(oracle.es_lower(q), rel=1e-6)
        if dist.xi_right < 1:
            assert dist.expected_shortfall(q, tail='right') == pytest.approx(oracle.es_upper(q), rel=1e-6)


def test_random_draws_follow_the_distribution():
    dist = ph.Phat(0.1, 1.0, 0.2, 0.3, s_left=1.2, s_right=0.9)
    x = dist.rvs(40_000, random_state=np.random.default_rng(7))
    assert st.kstest(x, dist.cdf).pvalue > 0.01
    assert np.mean(x < dist.junctions[0]) == pytest.approx(dist.tail_masses[0], abs=0.01)
    z = dist.std_rvs(40_000, random_state=np.random.default_rng(8))
    assert abs(np.mean(z)) < 0.05


def test_gaussian_share_grows_as_the_junction_moves_out():
    shares = [ph.Phat(0, 1, 0.2, 0.2, s_left=s, s_right=s).body_mass for s in (0.5, 1.0, 1.5, 2.0)]
    assert shares == sorted(shares)
    assert shares[0] < 0.3 and shares[-1] > 0.9


# ---- estimation -------------------------------------------------------------
TRUE = dict(mu=0.1, sigma=1.0, xi_left=0.2, xi_right=0.3, s_left=1.2, s_right=0.9)


@pytest.fixture(scope='module')
def sample():
    return ph.Phat(**TRUE).rvs(40_000, random_state=np.random.default_rng(11))


@pytest.fixture(scope='module')
def free_fit(sample):
    return ph.Phat.fit(sample, junction='free', per_side=True)


@pytest.fixture(scope='module')
def implied_fit(sample):
    return ph.Phat.fit(sample)


def test_free_fit_recovers_the_junctions(free_fit):
    assert free_fit.names == ('mu', 'sigma', 'xi_left', 'xi_right', 's_left', 's_right')
    assert free_fit.junction == 'free' and free_fit.per_side is True
    assert free_fit.converged
    tol = dict(mu=0.05, sigma=0.05, xi_left=0.06, xi_right=0.06, s_left=0.15, s_right=0.15)
    for name, truth in TRUE.items():
        assert abs(free_fit.params_dict[name] - truth) < tol[name], name


def test_free_fit_standard_errors_cover_the_truth(free_fit):
    assert free_fit.bse.shape == (6,)
    assert np.all(np.isfinite(free_fit.bse)) and np.all(free_fit.bse > 0)
    lo, hi = free_fit.conf_int(alpha=0.001).T
    for i, name in enumerate(free_fit.names):
        assert lo[i] < TRUE[name] < hi[i], name


def test_free_fit_result_builds_the_fitted_distribution(sample, free_fit):
    dist = free_fit.dist
    assert dist.junction == ('free', 'free')
    assert dist.s_left == pytest.approx(free_fit.params_dict['s_left'])
    assert free_fit.llf == pytest.approx(float(np.sum(dist.logpdf(sample))))
    assert set(free_fit.all_params) == {'mu', 'sigma', 'xi_left', 'xi_right', 's_left', 's_right'}
    assert free_fit.aic == pytest.approx(2 * 6 - 2 * free_fit.llf)
    for name in ('s_left', 's_right'):
        assert name in free_fit.summary()


def test_free_junction_beats_the_implied_one_on_free_junction_data(free_fit, implied_fit):
    assert implied_fit.names == ('mu', 'sigma', 'xi_left', 'xi_right')
    assert set(implied_fit.all_params) == {'mu', 'sigma', 'xi_left', 'xi_right'}
    assert implied_fit.dist.junction == ('implied', 'implied')
    assert implied_fit.junction == 'implied' and implied_fit.per_side is False
    assert free_fit.llf > implied_fit.llf + 100


def test_free_fit_on_implied_junction_data_finds_the_implied_junction():
    truth = ph.Phat(0.0, 1.0, 0.25, 0.25)
    x = truth.rvs(40_000, random_state=np.random.default_rng(3))
    res = ph.Phat.fit(x, junction='free', per_side=True)
    implied = ph.Phat.fit(x)
    assert res.llf >= implied.llf - 1e-6
    assert res.llf - implied.llf < 8          # two extra parameters, nothing to gain
    assert abs(res.params_dict['s_left'] - truth.s_left) < 0.12
    assert abs(res.params_dict['s_right'] - truth.s_right) < 0.12


def test_free_junction_fit_uses_one_distance_unless_per_side(sample):
    res = ph.Phat.fit(sample, junction='free')
    assert res.names == ('mu', 'sigma', 'xi_left', 'xi_right', 's')
    assert res.junction == 'free' and res.per_side is False
    dist = res.dist
    assert dist.s_left == dist.s_right == pytest.approx(res.params_dict['s'])
    assert res.all_params['s_left'] == res.all_params['s_right']
    assert 0.9 < res.params_dict['s'] < 1.2          # between the two true distances


def test_two_stage_fit_with_a_free_junction(sample):
    res = ph.Phat.fit(sample, xi_left=TRUE['xi_left'], xi_right=TRUE['xi_right'], junction='free',
                      per_side=True)
    assert res.names == ('mu', 'sigma', 's_left', 's_right')
    assert res.fixed['xi_left'] == TRUE['xi_left']
    for name in res.names:
        assert abs(res.params_dict[name] - TRUE[name]) < 0.1, name
    assert np.all(np.isfinite(res.bse))


def test_fit_with_given_junctions(sample):
    res = ph.Phat.fit(sample, s_left=TRUE['s_left'], s_right=TRUE['s_right'])
    assert res.names == ('mu', 'sigma', 'xi_left', 'xi_right')
    assert res.fixed['s_left'] == TRUE['s_left'] and res.fixed['s_right'] == TRUE['s_right']
    assert res.dist.s_left == TRUE['s_left']
    for name in res.names:
        assert abs(res.params_dict[name] - TRUE[name]) < 0.05, name


def test_fit_options_are_validated(sample):
    with pytest.raises(ValueError):
        ph.Phat.fit(sample, junction='moving')
    with pytest.raises(ValueError):
        ph.Phat.fit(sample, junction='symmetric')        # not a junction: use per_side
    with pytest.raises(ValueError):
        ph.Phat.fit(sample, per_side=True)               # needs junction='free'
    with pytest.raises(ValueError):
        ph.Phat.fit(sample, per_side=True, s_left=1.0, s_right=1.0)
    with pytest.raises(ValueError):
        ph.Phat.fit(sample, junction='free', s_left=1.0, s_right=1.0)
    with pytest.raises(ValueError):
        ph.Phat.fit(sample, s_left=1.0)
    with pytest.raises(ValueError):
        ph.Phat.fit(sample, junction='free', backend='statsmodels')


def test_fit_phat_function_takes_the_same_options(sample):
    res = fit_phat(sample[:5000], junction='free')
    assert res.names[-1] == 's' and res.per_side is False
    res = fit_phat(sample[:5000], junction='free', per_side=True)
    assert res.names[-2:] == ('s_left', 's_right') and res.per_side is True
