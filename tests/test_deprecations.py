"""
Names from phat-tails 0.0.x keep working for one release and warn.

New names: ``sigma`` (was ``sig``), ``xi_left`` / ``xi_right`` (``xi_l`` /
``xi_r``), ``weights`` (``p``), ``tail='left'|'right'`` (``rtail``),
``n_paths`` / ``horizon`` (``iters`` / ``periods``) and ``n_runs`` (``iters``
in the bootstrap).
"""
import pathlib
import warnings

import numpy as np
import pytest

import phat as ph
from phat.tseries import Garchcaster

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'student_t3_n1500.csv'


def _same(a: ph.Phat, b: ph.Phat) -> bool:
    return (a.mu, a.sigma, a.xi_left, a.xi_right) == (b.mu, b.sigma, b.xi_left, b.xi_right) \
        and np.array_equal(a.weights, b.weights)


class TestNewNamesAreSilent:
    def test_everything_by_new_keyword(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            dist = ph.Phat(mu=0.1, sigma=2.0, xi_left=0.2, xi_right=0.3, weights=(0.4, 0.6))
            assert (dist.sigma, dist.xi_left, dist.xi_right) == (2.0, 0.2, 0.3)
            assert dist.weights.tolist() == [0.4, 0.6]
            right = ph.CarBenRight(xi=0.2, mu=0.0, sigma=1.5)
            left = ph.CarBenHybrid(0.2, 0.0, sigma=1.5, tail='left')
            assert right.sigma == left.sigma == 1.5
            assert dist.params == {
                'mu': 0.1, 'sigma': 2.0, 'xi_left': 0.2, 'xi_right': 0.3,
                'a_left': dist.left.a, 'a_right': dist.right.a,
                'b_left': dist.left.b, 'b_right': dist.right.b,
            }
            assert dist.learnable_params == [0.1, 2.0, 0.2, 0.3]


class TestPhat:
    def test_old_keywords_warn_and_work(self):
        new = ph.Phat(0.1, 2.0, 0.2, 0.3, weights=(0.4, 0.6))
        with pytest.warns(DeprecationWarning, match='sigma'):
            old = ph.Phat(0.1, sig=2.0, xi_left=0.2, xi_right=0.3, weights=(0.4, 0.6))
        assert _same(old, new)
        with pytest.warns(DeprecationWarning, match='xi_left'):
            old = ph.Phat(0.1, 2.0, xi_l=0.2, xi_right=0.3, weights=(0.4, 0.6))
        assert _same(old, new)
        with pytest.warns(DeprecationWarning, match='xi_right'):
            old = ph.Phat(0.1, 2.0, 0.2, xi_r=0.3, weights=(0.4, 0.6))
        assert _same(old, new)
        with pytest.warns(DeprecationWarning, match='weights'):
            old = ph.Phat(0.1, 2.0, 0.2, 0.3, p=(0.4, 0.6))
        assert _same(old, new)

    def test_old_and_new_keyword_together_is_an_error(self):
        with pytest.raises(TypeError):
            ph.Phat(0.1, sig=2.0, sigma=2.0, xi_left=0.2, xi_right=0.3)

    def test_old_attributes_warn_and_work(self):
        dist = ph.Phat(0.1, 2.0, 0.2, 0.3, weights=(0.4, 0.6))
        with pytest.warns(DeprecationWarning, match='sigma'):
            assert dist.sig == 2.0
        with pytest.warns(DeprecationWarning, match='xi_left'):
            assert dist.xi_l == 0.2
        with pytest.warns(DeprecationWarning, match='xi_right'):
            assert dist.xi_r == 0.3
        with pytest.warns(DeprecationWarning, match='weights'):
            assert dist.p.tolist() == [0.4, 0.6]

    def test_old_param_keys_warn_and_work(self):
        params = ph.Phat(0.1, 2.0, 0.2, 0.3).params
        for old, new in [('sig', 'sigma'), ('xi_l', 'xi_left'), ('xi_r', 'xi_right'),
                         ('a_l', 'a_left'), ('a_r', 'a_right'), ('b_l', 'b_left'), ('b_r', 'b_right')]:
            with pytest.warns(DeprecationWarning, match=new):
                assert params[old] == params[new]

    def test_fit_takes_old_tail_keywords(self):
        x = ph.Phat(0, 1, 0.2, 0.3).rvs(2000, random_state=1)
        new = ph.Phat.fit(x, xi_left=0.2, xi_right=0.3)
        with pytest.warns(DeprecationWarning):
            old = ph.Phat.fit(x, xi_l=0.2, xi_r=0.3)
        assert np.array_equal(old.params, new.params)
        assert new.names == ('mu', 'sigma')
        assert set(new.fixed) == {'xi_left', 'xi_right'}
        assert ph.Phat.fit(x).names == ('mu', 'sigma', 'xi_left', 'xi_right')


class TestCarBen:
    @pytest.mark.parametrize('cls', [ph.CarBenRight, ph.CarBenLeft])
    def test_sig_keyword_and_attribute(self, cls):
        with pytest.warns(DeprecationWarning, match='sigma'):
            comp = cls(0.2, 0.0, sig=1.5)
        assert comp.sigma == 1.5
        with pytest.warns(DeprecationWarning, match='sigma'):
            assert comp.sig == 1.5

    def test_hybrid_sig_keyword(self):
        with pytest.warns(DeprecationWarning, match='sigma'):
            comp = ph.CarBenHybrid(0.2, 0.0, sig=1.5)
        assert isinstance(comp, ph.CarBenRight) and comp.sigma == 1.5

    def test_rtail_keyword(self):
        with pytest.warns(DeprecationWarning, match="tail='left'"):
            comp = ph.CarBenHybrid(0.2, 0.0, 1.5, rtail=False)
        assert isinstance(comp, ph.CarBenLeft) and comp.xi == 0.2
        with pytest.warns(DeprecationWarning, match="tail='right'"):
            comp = ph.CarBenHybrid(0.2, 0.0, 1.5, rtail=True)
        assert isinstance(comp, ph.CarBenRight)

    def test_rtail_as_fifth_positional_boolean(self):
        with pytest.warns(DeprecationWarning, match="tail='left'"):
            comp = ph.CarBenHybrid(0.2, 0.0, 1.5, None, False)
        assert isinstance(comp, ph.CarBenLeft)

    def test_rtail_given_means_xi_is_the_shape(self):
        # the sign shim applies only when no side is given at all
        with pytest.warns(DeprecationWarning):
            comp = ph.CarBenHybrid(-0.2, 0.0, 1.5, rtail=True)
        assert isinstance(comp, ph.CarBenRight) and comp.xi == -0.2

    def test_tail_and_rtail_together_is_an_error(self):
        with pytest.raises(TypeError):
            ph.CarBenHybrid(0.2, 0.0, 1.5, tail='left', rtail=False)


class TestPhatStack:
    def test_p_keyword_and_attribute(self):
        d1, d2 = ph.Phat(0, 1, 0.2, 0.2), ph.Phat(0, 2, 0.3, 0.3)
        with pytest.warns(DeprecationWarning, match='weights'):
            stack = ph.PhatStack(d1, d2, p=[0.25, 0.75])
        assert stack.weights.tolist() == [0.25, 0.75]
        with pytest.warns(DeprecationWarning, match='weights'):
            assert stack.p.tolist() == [0.25, 0.75]


class TestGarchcaster:
    KW = dict(garch=[0.1, 0.1, 0.8], arma=[0.05], y=np.array([0.5, -0.3]),
              vols=np.array([1.1, 1.2]), resids=np.array([0.4, -0.6]), use_backcast=False)

    def test_constructor_keywords_and_attributes(self):
        new = Garchcaster(**self.KW, n_paths=4, horizon=5)
        assert (new.n_paths, new.horizon) == (4, 5)
        with pytest.warns(DeprecationWarning, match='n_paths'):
            old = Garchcaster(**self.KW, iters=4, horizon=5)
        with pytest.warns(DeprecationWarning, match='horizon'):
            old = Garchcaster(**self.KW, n_paths=4, periods=5)
        assert (old.n_paths, old.horizon) == (4, 5)
        with pytest.warns(DeprecationWarning, match='n_paths'):
            assert old.iters == 4
        with pytest.warns(DeprecationWarning, match='horizon'):
            assert old.periods == 5

    def test_forecast_keywords_and_result_attributes(self):
        cast = Garchcaster(**self.KW, n_paths=4, horizon=5)
        new = cast.forecast(n_paths=3, horizon=6, seed=1)
        assert new.values.shape == (3, 6)
        assert (new.n_paths, new.horizon) == (3, 6)
        with pytest.warns(DeprecationWarning):
            old = cast.forecast(iters=3, periods=6, seed=1)
        assert np.array_equal(old.values, new.values)
        with pytest.warns(DeprecationWarning, match='n_paths'):
            assert new.iters == 3
        with pytest.warns(DeprecationWarning, match='horizon'):
            assert new.periods == 6


def test_bootstrap_iters_keyword():
    from phat.bootstrap import two_tailed_hill_double_bootstrap
    x = np.loadtxt(FIXTURE)
    kw = dict(seed=1, pbar_kwargs={'disable': True})
    new = two_tailed_hill_double_bootstrap(x, n_runs=2, **kw)
    with pytest.warns(DeprecationWarning, match='n_runs'):
        old = two_tailed_hill_double_bootstrap(x, iters=2, **kw)
    assert old == new


def test_warnings_point_at_the_caller():
    # A warning attributed to a phat module would be hidden from users by
    # Python's default filters (and turned into an error by this test suite).
    dist = ph.Phat(0.1, 2.0, 0.2, 0.3)
    calls = [
        lambda: ph.Phat(0.1, sig=2.0, xi_left=0.2, xi_right=0.3),
        lambda: ph.CarBenRight(0.2, 0.0, sig=1.5),
        lambda: ph.CarBenHybrid(0.2, 0.0, 1.5, rtail=False),
        lambda: ph.CarBenHybrid(-0.2, 0.0, 1.5),
        lambda: dist.sig,
        lambda: dist.params['xi_l'],
        lambda: ph.Phat.fit(dist.rvs(500, random_state=1), xi_l=0.2, xi_r=0.3),
        lambda: Garchcaster(**TestGarchcaster.KW, iters=2, periods=3),
    ]
    for call in calls:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            call()
        assert caught, call
        assert all(w.filename == __file__ for w in caught), [(w.filename, str(w.message)) for w in caught]
