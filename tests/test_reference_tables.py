"""
Reference tables from the professionalisation brief (2026-11-25), produced by
an independent NumPy/SciPy implementation (quadrature + root-finding).

Package values before the fix are recorded in the comments so the change is
visible in the history.
"""
import numpy as np
import pytest
import scipy.stats as st

import phat as ph

# Phat(mu, sig, xi_l, xi_r): true mean, true variance.
# Package before fix: (0.0796, 3.7927^2), (0.1427, 0.7572^2), (0.0, 3.4837^2), (3.3704, 6.9869^2)
MOMENTS = [
    ((0, 1, 0.2, 0.25), 0.0778, 24.0587),
    ((0.1, 0.2, 0.2, 0.25), 0.1156, 0.9623),
    ((0, 1, 0.2, 0.2), 0.0000, 21.1855),
    ((2, 2, 0.2, 0.2), 2.0000, 84.7421),
]

# Quantiles of Phat(0, 1, 0.2, 0.25).
# Package before fix: -19.5545, -10.0134, -5.3410, -1.6374, 0.0057, 1.6801, 5.6841, 11.1669, 23.5767
QUANTILES = [
    (0.001, -29.7649),
    (0.01, -13.8558),
    (0.05, -6.3699),
    (0.25, -1.2999),
    (0.5, 0.0020),
    (0.75, 1.3112),
    (0.95, 6.7030),
    (0.99, 15.2973),
    (0.999, 35.4893),
]

# Symmetric xi, mu = 0, sig = 1: s (junction in sigma units), P(X > a_r), P(a_l < X < a_r)
JUNCTIONS = [
    (1e-9, 0.372, 0.412, 0.176),
    (0.10, 0.404, 0.405, 0.190),
    (0.20, 0.435, 0.399, 0.202),
    (0.25, 0.451, 0.396, 0.208),
    (0.50, 0.522, 0.383, 0.234),
    (1.00, 0.647, 0.361, 0.277),
]


@pytest.mark.parametrize("args, mean, var", MOMENTS)
def test_moments_match_reference(args, mean, var):
    dist = ph.Phat(*args)
    assert dist.mean() == pytest.approx(mean, abs=5e-5)
    assert dist.var() == pytest.approx(var, abs=5e-5)
    assert dist.std() == pytest.approx(np.sqrt(var), abs=5e-5)


@pytest.mark.parametrize("q, x", QUANTILES)
def test_quantiles_match_reference(q, x):
    dist = ph.Phat(0, 1, 0.2, 0.25)
    assert dist.ppf(q) == pytest.approx(x, abs=5e-5)


def test_quantiles_vectorised():
    dist = ph.Phat(0, 1, 0.2, 0.25)
    q = np.array([row[0] for row in QUANTILES])
    x = np.array([row[1] for row in QUANTILES])
    np.testing.assert_allclose(dist.ppf(q), x, atol=5e-5)
    np.testing.assert_allclose(dist.ppf(q.reshape(3, 3)), x.reshape(3, 3), atol=5e-5)


@pytest.mark.parametrize("xi, s, tail, body", JUNCTIONS)
def test_junctions_and_masses(xi, s, tail, body):
    dist = ph.Phat(0, 1, xi, xi)
    a_l, a_r = dist.junctions
    assert a_r == pytest.approx(s, abs=5e-4)
    assert a_l == pytest.approx(-s, abs=5e-4)
    left_mass, right_mass = dist.tail_masses
    assert right_mass == pytest.approx(tail, abs=5e-4)
    assert left_mass == pytest.approx(tail, abs=5e-4)
    assert dist.body_mass == pytest.approx(body, abs=5e-4)
    # the accessors agree with the CDF
    assert dist.sf(a_r) == pytest.approx(right_mass, abs=1e-12)
    assert dist.cdf(a_r) - dist.cdf(a_l) == pytest.approx(dist.body_mass, abs=1e-12)


def test_sampler_passes_ks_against_own_cdf():
    # Package before fix: KS distance 0.044-0.046 at n = 200,000.
    dist = ph.Phat(0, 1, 0.2, 0.25)
    x = dist.rvs(size=200_000, random_state=np.random.default_rng(0))
    assert x.shape == (200_000,)
    res = st.kstest(x, dist.cdf)
    assert res.statistic < 0.005
    assert res.pvalue > 0.01


def test_sampler_accepts_generator_int_and_none():
    dist = ph.Phat(0, 1, 0.2, 0.25)
    a = dist.rvs(size=5, random_state=np.random.default_rng(42))
    b = dist.rvs(size=5, random_state=42)
    np.testing.assert_array_equal(a, b)
    assert np.isscalar(dist.rvs()) or np.ndim(dist.rvs()) == 0
    assert dist.rvs(size=(3, 4)).shape == (3, 4)


class TestCarBenHybridConstructor:
    """
    ``xi`` is the GPD shape of the tail; the side comes from ``rtail`` (or from
    using ``CarBenRight`` / ``CarBenLeft`` directly). Until 0.0.x a negative
    ``xi`` selected the left tail; that reading survives for one release, with
    a ``FutureWarning``, only when ``rtail`` is not given.
    """

    def test_positive_xi_defaults_to_right_without_warning(self, recwarn):
        obj = ph.CarBenHybrid(0.25, 1, 2)
        assert isinstance(obj, ph.CarBenRight)
        assert obj.xi == 0.25
        assert not recwarn.list

    def test_keyword_positive_xi_gives_right(self):
        obj = ph.CarBenHybrid(xi=0.25, mu=1, sig=2)
        assert isinstance(obj, ph.CarBenRight)
        assert obj.xi == 0.25

    def test_rtail_false_gives_left(self, recwarn):
        obj = ph.CarBenHybrid(0.25, 1, 2, rtail=False)
        assert isinstance(obj, ph.CarBenLeft)
        assert obj.xi == 0.25
        assert not recwarn.list

    def test_rtail_as_fifth_positional_argument(self):
        obj = ph.CarBenHybrid(0.25, 1, 2, None, False)
        assert isinstance(obj, ph.CarBenLeft)

    # ---- new convention: with `rtail` given, xi is the shape, whatever its sign
    def test_negative_xi_with_rtail_true_is_a_bounded_right_tail(self, recwarn):
        obj = ph.CarBenHybrid(-0.25, 1, 2, rtail=True)
        assert isinstance(obj, ph.CarBenRight)
        assert obj.xi == -0.25
        endpoint = obj.a + obj.b / 0.25
        assert obj.sf(endpoint + 1e-9) == 0 and obj.sf(endpoint - 1e-3) > 0
        assert not recwarn.list

    def test_negative_xi_with_rtail_false_is_a_bounded_left_tail(self, recwarn):
        obj = ph.CarBenHybrid(-0.25, 1, 2, rtail=False)
        assert isinstance(obj, ph.CarBenLeft)
        assert obj.xi == -0.25
        endpoint = obj.a - obj.b / 0.25
        assert obj.cdf(endpoint - 1e-9) == 0 and obj.cdf(endpoint + 1e-3) > 0
        assert not recwarn.list

    def test_shape_at_or_below_minus_one_is_rejected(self):
        with pytest.raises(ValueError):
            ph.CarBenHybrid(-1.0, 0, 1, rtail=True)

    # ---- one-release shim: negative xi without `rtail` still means "left tail"
    def test_negative_xi_without_rtail_warns_and_gives_left(self):
        with pytest.warns(FutureWarning, match='rtail'):
            obj = ph.CarBenHybrid(-0.25, 1, 2)
        assert isinstance(obj, ph.CarBenLeft)
        assert obj.xi == 0.25

    def test_keyword_negative_xi_without_rtail_warns_and_gives_left(self):
        with pytest.warns(FutureWarning, match='rtail'):
            obj = ph.CarBenHybrid(xi=-0.25, mu=1, sig=2)
        assert isinstance(obj, ph.CarBenLeft)
        assert obj.xi == 0.25

    def test_integer_xi_accepted(self):
        assert isinstance(ph.CarBenHybrid(1, 0, 1), ph.CarBenRight)
        assert isinstance(ph.CarBenHybrid(1, 0, 1, rtail=False), ph.CarBenLeft)
