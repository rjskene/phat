import pytest
import numpy as np
import phat as ph

n = 1052
days = 30

class TestPhatStack:

    @pytest.fixture
    def pstack(self):
        mu, sig, xil, xir = 4, .25, .3, .35
        phat1 = ph.Phat(mu, sig, xil, xir)

        mu, sig, xil, xir = -2, .35, .25, .45
        phat2 = ph.Phat(mu, sig, xil, xir)

        mu, sig, xil, xir = -1, .9, .45, .2
        phat3 = ph.Phat(mu, sig, xil, xir)

        p = np.array([.22,.5,.28])

        return ph.PhatStack(phat1, phat2, phat3, p=p)

    @pytest.fixture
    def n(self):
        return 1052

    @pytest.fixture
    def days(self):
        return 30

    def test_correct_size(self, pstack, n, days):
        rvs = pstack.rvs((n, days))
    
        assert rvs.shape[0] == n
        assert rvs.shape[1] == days

    def test_return_splits(self, pstack, n, days):
        rvs = pstack.rvs((n, days), return_splits=True)

        assert len(rvs) == 2
        assert rvs[1].sum() == n
        assert np.isclose(pstack.p, rvs[1] / rvs[1].sum(), rtol=.01).all()
