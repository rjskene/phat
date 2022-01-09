"""test_class_parametrization.py"""
import pytest
import numpy as np
import phat as ph

@pytest.mark.parametrize('cb', ['cbr', 'cbl'], indirect=True)
class TestGroup:
    """A class with common parameters, `param1` and `param2`."""

    @pytest.fixture
    def cb(self, request):
        return request.getfixturevalue(request.param)

    @pytest.fixture
    def r_params(self):
        return (.25, 1, 2)

    @pytest.fixture
    def l_params(self):
        return (-.25, 1, 2)
    
    @pytest.fixture
    def cbr(self, r_params):
        return ph.CarBenHybrid(*r_params)
    
    @pytest.fixture
    def cbl(self, l_params):
        return ph.CarBenHybrid(*l_params)

    def is_right(self, cb):
	    return isinstance(cb, ph.dists.CarBenRight)

    def test_correct_params(self, cb, r_params, l_params):
        params = r_params if self.is_right(cb) else l_params
        assert cb.xi == params[0] if self.is_right(cb) else -params[0]
        assert cb.mu == params[1]
        assert cb.sig == params[2]

    def test_correct_body_params(self, cb, l_params):
        assert cb.body.args == l_params[1:3]

    def test_rtail_keyword_for_left(self, cb, r_params):
        if self.is_right(cb):
            carb = ph.CarBenHybrid(*r_params, rtail=False)
            assert isinstance(carb, ph.dists.CarBenLeft)
            assert carb.mu == cb.mu
            assert carb.sig == cb.sig
            assert carb.xi == cb.xi

    def test_gamma(self, cb):
        if self.is_right(cb):
            assert np.isclose(cb.sf(cb.a), 1 / cb.gamma)
        else:
            assert cb.cdf(cb.a) == 1 / cb.gamma

    def test_cdf(self, cb):
        if self.is_right(cb):
            assert np.isclose(cb.cdf(cb.a), 1 - 1 / cb.gamma)
        else:
            assert np.isclose(cb.cdf(cb.a), 1 / cb.gamma)
            assert cb.cdf(cb.a) + (cb.body.cdf(np.inf) - cb.body.cdf(cb.a)) / cb.gamma == 1
            assert (1/cb.gamma) + (cb.body.cdf(np.inf) - cb.body.cdf(cb.a)) / cb.gamma == 1
            assert 1 + (cb.body.cdf(np.inf) - cb.body.cdf(cb.a)) == cb.gamma
            x = 3
            assert np.isclose((1 - cb.body.cdf(cb.a) + cb.body.cdf(x)) / cb.gamma, cb.cdf(x)[0])
            assert np.isclose(cb.body.cdf(cb.a), cb.gamma*(2*cb.qjunc - 1))

    def test_sf(self, cb):
        vals = [-1,0.02,2.5]
        assert np.all(cb.sf(vals) == 1 - cb.cdf(vals))

    def test_rvs(self, cb):
        size = 1
        assert cb.rvs(size).shape == (1,)
        size = [2]
        assert cb.rvs(size).shape == (2,)
        size = (3,2)
        assert cb.rvs(size).shape == size
        seed = 1234
        assert np.all(cb.rvs(size, seed=seed) == cb.rvs(size, seed=seed))

    def test_qjunc(self, cb):
        assert isinstance(cb.qjunc, float)
        if self.is_right(cb):
            assert cb.qjunc == 1 - (1/cb.gamma)
            assert np.isclose(cb.qjunc, 0.4025721108239665)
        else:
            assert cb.qjunc == 1/cb.gamma
            assert np.isclose(cb.qjunc, 0.5974278891760335)

    def test_ppf(self, cb):
        assert np.isclose(cb.ppf(cb.qjunc), cb.a)

        q = [.1,.9]
        if self.is_right(cb):
            assert np.isclose(cb.ppf(q[0]), cb.body.ppf(cb.gamma*q[0]))
            assert np.isclose(cb.ppf(q[1]), cb.tail.ppf(cb.gamma*q[1] - cb.gamma*cb.qjunc))
        else:
            assert np.isclose(cb.ppf(q[1]), cb.body.ppf(cb.gamma*q[1] + cb.gamma*cb.qjunc - cb.gamma))
            assert np.isclose(cb.ppf(q[0]), -cb.tail.ppf(1 - cb.gamma*q[0]))
