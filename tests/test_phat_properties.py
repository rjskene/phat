"""Mathematical properties of Phat checked against the independent oracle."""
import numpy as np
import pytest
import scipy.stats as st
from scipy.integrate import quad

import phat as ph
from tests.oracle import OraclePhat

CASES = [
    (0, 1, 0.2, 0.25),
    (0.1, 0.2, 0.2, 0.25),
    (2, 2, 0.2, 0.2),
    (-1.5, 0.7, 0.0, 0.45),
    (0.3, 3.0, 0.6, 0.05),
]


@pytest.fixture(params=CASES, ids=lambda c: f"Phat{c}")
def pair(request):
    return ph.Phat(*request.param), OraclePhat(*request.param)


def test_density_integrates_to_one(pair):
    dist, oracle = pair
    total = oracle._integrate(lambda x: float(dist.pdf(x)))
    assert total == pytest.approx(1.0, abs=1e-7)


def test_pdf_cdf_match_oracle(pair):
    dist, oracle = pair
    x = np.linspace(dist.mu - 8 * dist.sig, dist.mu + 8 * dist.sig, 201)
    np.testing.assert_allclose(dist.pdf(x), oracle.pdf(x), rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(dist.cdf(x), oracle.cdf(x), rtol=1e-12)


def test_cdf_monotone_with_limits(pair):
    dist, _ = pair
    x = np.linspace(dist.mu - 50 * dist.sig, dist.mu + 50 * dist.sig, 2001)
    c = dist.cdf(x)
    assert np.all(np.diff(c) >= 0)
    assert dist.cdf(-np.inf) == 0.0
    assert dist.cdf(np.inf) == 1.0
    assert dist.sf(-np.inf) == 1.0
    assert dist.sf(np.inf) == 0.0


def test_ppf_inverts_cdf(pair):
    dist, _ = pair
    q = np.array([1e-6, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-4, 1 - 1e-6])
    x = dist.ppf(q)
    np.testing.assert_allclose(dist.cdf(x), q, rtol=1e-9, atol=1e-14)
    xs = np.linspace(dist.mu - 6 * dist.sig, dist.mu + 6 * dist.sig, 31)
    np.testing.assert_allclose(dist.ppf(dist.cdf(xs)), xs, rtol=1e-9, atol=1e-9)


def test_ppf_matches_oracle(pair):
    dist, oracle = pair
    q = np.array([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    np.testing.assert_allclose(dist.ppf(q), oracle.ppf(q), rtol=1e-9, atol=1e-9)


def test_ppf_edge_values():
    dist = ph.Phat(0, 1, 0.2, 0.25)
    assert dist.ppf(0) == -np.inf
    assert dist.ppf(1) == np.inf
    assert np.isnan(dist.ppf(-0.1))
    assert np.isnan(dist.ppf(1.1))
    assert np.isnan(dist.ppf(np.nan))


def test_isf_is_upper_quantile(pair):
    dist, _ = pair
    p = np.array([1e-8, 1e-5, 0.01, 0.2, 0.5, 0.8, 0.99])
    x = dist.isf(p)
    np.testing.assert_allclose(dist.sf(x), p, rtol=1e-9)
    np.testing.assert_allclose(dist.isf(0.3), dist.ppf(0.7), rtol=1e-10)


def test_sf_accurate_in_far_tail(pair):
    dist, oracle = pair
    # 1 - cdf(x) is exactly 0 out here; sf must still carry the GPD tail.
    x = dist.mu + dist.sig * np.array([1e3, 1e5, 1e7])
    sf = dist.sf(x)
    assert np.all(sf > 0)
    assert np.all(np.diff(sf) < 0)
    # right component tail, closed form via the oracle's component dictionary
    R = oracle.R
    expected = dist.p[1] * st.genpareto.sf(x, R["xi"], loc=oracle.a_r, scale=R["b"]) / R["gam"]
    # the left component adds only a Gaussian remnant, negligible out here
    np.testing.assert_allclose(sf, expected, rtol=1e-10)
    xl = dist.mu - dist.sig * np.array([1e3, 1e5, 1e7])
    cdf = dist.cdf(xl)
    L = oracle.L
    expected_l = dist.p[0] * st.genpareto.sf(2 * dist.mu - xl, L["xi"], loc=oracle.a_r if False else dist.mu + dist.sig * L["s"], scale=L["b"]) / L["gam"]
    np.testing.assert_allclose(cdf, expected_l, rtol=1e-10)


def test_log_functions(pair):
    dist, _ = pair
    x = np.linspace(dist.mu - 30 * dist.sig, dist.mu + 30 * dist.sig, 301)
    np.testing.assert_allclose(dist.logpdf(x), np.log(dist.pdf(x)), rtol=1e-12)
    np.testing.assert_allclose(dist.logcdf(x), np.log(dist.cdf(x)), rtol=1e-12)
    np.testing.assert_allclose(dist.logsf(x), np.log(dist.sf(x)), rtol=1e-12)
    far = dist.mu + dist.sig * 1e12
    assert np.isfinite(dist.logsf(far))
    assert np.isfinite(dist.logpdf(far))
    assert np.isfinite(dist.logcdf(-far))
    np.testing.assert_allclose(dist.nll(x), -dist.logpdf(x))


def test_density_and_derivative_continuous_at_junctions(pair):
    dist, _ = pair
    h = 1e-6 * dist.sig
    for a in dist.junctions:
        left, right = dist.pdf(a - h), dist.pdf(a + h)
        assert left == pytest.approx(right, rel=1e-5)
        d_left = (dist.pdf(a) - dist.pdf(a - h)) / h
        d_right = (dist.pdf(a + h) - dist.pdf(a)) / h
        assert d_left == pytest.approx(d_right, rel=1e-4, abs=1e-8)


def test_moments_match_quadrature(pair):
    dist, oracle = pair
    if max(dist.xi_l, dist.xi_r) < 0.5:
        assert dist.mean() == pytest.approx(oracle.mean(), rel=1e-7, abs=1e-9)
        assert dist.var() == pytest.approx(oracle.var(), rel=1e-7)
        m, v = dist.stats()
        assert (m, v) == (dist.mean(), dist.var())
    else:
        assert dist.mean() == pytest.approx(oracle.mean(), rel=1e-7, abs=1e-9)
        assert dist.var() == np.inf


def test_component_moments_with_location_and_scale():
    # The old code passed unstandardised bounds to truncnorm.
    for mu, sig, xi in [(0, 1, 0.2), (2, 2, 0.2), (-1, 0.3, 0.0)]:
        right = ph.CarBenRight(xi, mu, sig)
        left = ph.CarBenLeft(xi, mu, sig)
        a = right.a

        def m(k, f):
            return quad(lambda x: x**k * float(f(x)), -np.inf, a, limit=400)[0] + quad(
                lambda x: x**k * float(f(x)), a, np.inf, limit=400
            )[0]

        assert right.mean() == pytest.approx(m(1, right.pdf), rel=1e-7)
        assert right.var() == pytest.approx(m(2, right.pdf) - m(1, right.pdf) ** 2, rel=1e-7)
        assert left.mean() == pytest.approx(2 * mu - right.mean(), rel=1e-12)
        assert left.var() == pytest.approx(right.var(), rel=1e-12)


@pytest.mark.parametrize(
    "xi_l, xi_r, mean, var",
    [
        (0.2, 1.0, np.inf, np.inf),
        (1.0, 0.2, -np.inf, np.inf),
        (1.0, 1.5, np.nan, np.inf),
        (0.5, 0.2, 0.0, np.inf),
        (0.2, 0.7, None, np.inf),
    ],
)
def test_moment_existence(xi_l, xi_r, mean, var):
    dist = ph.Phat(0, 1, xi_l, xi_r)
    got_mean = dist.mean()
    if mean is None:
        assert np.isfinite(got_mean)
    elif np.isnan(mean):
        assert np.isnan(got_mean)
    else:
        assert got_mean == mean if np.isinf(mean) else got_mean == pytest.approx(mean, abs=1e-12)
    assert dist.var() == var
    assert dist.std() == var


def test_location_scale_family():
    base = ph.Phat(0, 1, 0.2, 0.25)
    mu, sig = 1.3, 2.5
    dist = ph.Phat(mu, sig, 0.2, 0.25)
    q = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
    np.testing.assert_allclose(dist.ppf(q), mu + sig * base.ppf(q), rtol=1e-10)
    z = np.linspace(-6, 6, 25)
    np.testing.assert_allclose(dist.pdf(mu + sig * z), base.pdf(z) / sig, rtol=1e-12)
    np.testing.assert_allclose(dist.cdf(mu + sig * z), base.cdf(z), rtol=1e-12)
    assert dist.mean() == pytest.approx(mu + sig * base.mean(), rel=1e-12)
    assert dist.var() == pytest.approx(sig**2 * base.var(), rel=1e-12)


def test_expected_shortfall_matches_quadrature(pair):
    dist, oracle = pair
    for q in (0.01, 0.05, 0.25):
        assert dist.expected_shortfall(q, tail="left") == pytest.approx(oracle.es_lower(q), rel=1e-7, abs=1e-9)
        if dist.xi_r < 1:
            assert dist.expected_shortfall(q, tail="right") == pytest.approx(oracle.es_upper(q), rel=1e-7, abs=1e-9)
    np.testing.assert_allclose(
        dist.expected_shortfall(np.array([0.01, 0.05])),
        [dist.expected_shortfall(0.01), dist.expected_shortfall(0.05)],
    )


def test_expected_shortfall_infinite_for_heavy_right_tail():
    dist = ph.Phat(0, 1, 0.2, 1.2)
    assert dist.expected_shortfall(0.05, tail="right") == np.inf
    assert np.isfinite(dist.expected_shortfall(0.05, tail="left"))


def test_mixture_weight_p():
    dist = ph.Phat(0, 1, 0.2, 0.25, p=(0.3, 0.7))
    oracle = OraclePhat(0, 1, 0.2, 0.25, p=(0.3, 0.7))
    np.testing.assert_array_equal(dist.p, [0.3, 0.7])
    x = np.linspace(-5, 5, 51)
    np.testing.assert_allclose(dist.pdf(x), oracle.pdf(x), rtol=1e-12)
    np.testing.assert_allclose(dist.ppf([0.01, 0.5, 0.99]), oracle.ppf([0.01, 0.5, 0.99]), rtol=1e-9)
    assert dist.mean() == pytest.approx(oracle.mean(), rel=1e-7)
    assert dist.var() == pytest.approx(oracle.var(), rel=1e-7)
    x = dist.rvs(100_000, random_state=1)
    assert st.kstest(x, dist.cdf).pvalue > 0.01
    with pytest.raises(ValueError):
        ph.Phat(0, 1, 0.2, 0.25, p=(0.3, 0.6))


def test_parameter_validation():
    with pytest.raises(ValueError):
        ph.Phat(0, -1, 0.2, 0.25)
    with pytest.raises(ValueError):
        ph.Phat(0, 1, -0.2, 0.25)
    with pytest.raises(ValueError):
        ph.Phat(0, 1, 0.2, -0.25)


def test_scalar_in_scalar_out():
    dist = ph.Phat(0, 1, 0.2, 0.25)
    for method in (dist.pdf, dist.cdf, dist.sf, dist.logpdf, dist.ppf, dist.isf):
        out = method(0.3)
        assert np.ndim(out) == 0
        assert isinstance(dist.pdf([0.3]), np.ndarray)
        assert dist.pdf([0.3]).shape == (1,)


def test_std_rvs_has_unit_scale():
    dist = ph.Phat(1, 2, 0.1, 0.1)
    x = dist.std_rvs(size=400_000, random_state=3)
    assert x.std() == pytest.approx(1.0, abs=0.02)


def test_phatstack_rvs_splits_and_regimes():
    p1, p2 = ph.Phat(4, 0.25, 0.3, 0.35), ph.Phat(-2, 0.35, 0.25, 0.45)
    stack = ph.PhatStack(p1, p2, p=[0.25, 0.75])
    y, splits = stack.rvs((1000, 30), random_state=0, return_splits=True)
    assert y.shape == (1000, 30)
    assert splits.sum() == 1000
    assert abs(splits[0] - 250) <= 1
    # first block comes from p1 (around 4), second from p2 (around -2)
    assert y[: splits[0]].mean() > 2
    assert y[splits[0] :].mean() < 0
    a = stack.rvs((10, 3), random_state=7)
    b = stack.rvs((10, 3), random_state=7)
    np.testing.assert_array_equal(a, b)
