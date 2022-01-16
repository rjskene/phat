import pytest
import warnings
import pickle
import numpy as np
import pandas as pd
import arch
import phat as ph

class TestGarchcast:
    
    @pytest.fixture
    def n(self):
        return 1052

    @pytest.fixture
    def days(self):
        return 90

    @pytest.fixture
    def yelp_rets(self):
        return pd.read_csv('yelp_returns.csv', index_col='Date')

    @pytest.fixture
    def garch_cmean(self):
        with open('yelp_garch_cmean.pickle', "rb") as output_file:
            res = pickle.load(output_file)
        return res

    @pytest.fixture
    def gcast_cmean(self):
        with open('cmean_cast.pickle', 'rb') as file:
            gcast = pickle.load(file)
        return gcast

    @pytest.fixture
    def cmean_phat(self):
        with open('cmean_phat.pickle', 'rb') as file:
            phat = pickle.load(file)
        return phat

    @pytest.fixture
    def cmean_cast(self):
        with open('cmean_foreres.pickle', 'rb') as file:
            fore = pickle.load(file)
        return fore

    @pytest.fixture
    def argarch211(self):
        with open('argarch211.pickle', "rb") as output_file:
            argarch = pickle.load(output_file)
        return argarch

    @pytest.fixture
    def argarch211_gcast(self):
        with open('argarch211_gcast.pickle', 'rb') as file:
            gcast = pickle.load(file)
        return gcast

    @pytest.fixture
    def argarch211_phat(self):
        with open('argarch211_phat.pickle', 'rb') as file:
            phat = pickle.load(file)
        return phat

    @pytest.fixture
    def argarch211_res(self):
        with open('argarch211_res.pickle', 'rb') as file:
            fore = pickle.load(file)
        return fore

    def test_infer_order(self, garch_cmean):
        from phat.tseries import PROCESSMIXIN
        om = PROCESSMIXIN()
        order = om._infer_order(garch_cmean.params)
        assert order == (2,2)

        order = om._infer_order(garch_cmean.params[1:])
        assert order == (2,1)

        order = om._infer_order(garch_cmean.params[2:])
        assert order == (1,1)

        order = om._infer_order(garch_cmean.params[:1])
        assert order == (1,0)

        order = om._infer_order(garch_cmean.params[:0])
        assert order == (0,0)

    def test_cmean_container(self, garch_cmean):
        from phat.tseries import MEAN

        mean = MEAN(params=garch_cmean.params[:1])
        
        assert np.isclose(mean.constant, 0.140971546937745)
        assert mean.m == 0
        assert mean.n == 0
        assert mean.order == (0,0)
        assert mean.max_order == 0
        assert not mean.ar_params.any()
        assert not mean.ma_params.any()
        assert np.isclose(mean.props[0], mean.properties[0])

    def test_cmeanvol_container(self, garch_cmean):
        from phat.tseries import VOL

        vol = VOL(params=garch_cmean.params[1:])
        
        assert np.isclose(vol.constant, 4.1518386536911125, rtol=.000001)
        assert vol.p == 1
        assert vol.q == 1
        assert vol.order == (1,1)
        assert vol.max_order == 1
        assert np.isclose(vol.arch_params[0], 0.14239334923253452, rtol=.000001)
        assert np.isclose(vol.garch_params[0], 0.5658305933665543)    

    def test_cmean_instantiation(self, garch_cmean, n, days):
        cast = ph.Garchcaster(
            garch=garch_cmean,
            iters=n,
            periods=days,
        )
        assert cast.order == (0,0,1,1)
        assert cast.mean.m == 0
        assert cast.mean.n == 0
        assert cast.vol.p == 1
        assert cast.vol.q == 1
        assert cast.maxlag == 1
        assert cast.iters == n
        assert cast.periods == days
        assert cast.dist is None
        assert cast.y_hist.size == 1
        assert np.isclose(cast.y_hist[0], -0.59065254, rtol=.00001).all()
        assert cast.vols_hist.size == 1
        assert np.isclose(cast.vols_hist[0], 3.47872169, rtol=.00001).all()
        assert cast.resids_hist.size == 1
        assert np.isclose(cast.resids_hist[0], 21.12120221, rtol=.00001).all()

    def test_order_warning(self, garch_cmean, n, days):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cast = ph.Garchcaster(
                garch=garch_cmean,
                iters=n,
                periods=days,
                order=(2,0,1,1)
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert 'OVERRIDDEN' in str(w[-1].message)

    # def test_cmean_forecast(self, garch_cmean, n, days):
    #     cast = ph.Garchcaster(
    #         garch=garch_cmean,
    #         iters=n,
    #         periods=days,
    #     )
    #     fore = cast.forecast()

    def test_argarch_container(self, argarch211_res):
        from phat.tseries import MEAN
        mean = MEAN(params=argarch211_res.params[:3], order=(2,0))
        assert np.isclose(mean.constant, 0.154055)
        assert np.isclose(mean.ar_params[0], -0.04711448)
        assert np.isclose(mean.ar_params[1], 0.0130214) 
        assert mean.m == 2
        assert mean.n == 0
        assert mean.order == (2,0)
        assert mean.max_order == 2
        assert mean.ar_params.any()
        assert not mean.ma_params.any()
        
    def test_argarch211vol_container(self, argarch211_res):
        from phat.tseries import VOL

        vol = VOL(params=argarch211_res.params[3:], order=(1,1))
        print (argarch211_res.params)
        assert vol.p == 1
        assert vol.q == 1
        assert vol.order == (1,1)
        assert vol.max_order == 1
        assert np.isclose(vol.constant, 4.440505, rtol=.000001)
        print (vol.arch_params)
        assert np.isclose(vol.arch_params[0], 0.151584, rtol=.0001)
        assert np.isclose(vol.garch_params[0], 0.536940, rtol=.0001)

    def test_cmean_instantiation(self, argarch211_res, n, days):
        cast = ph.Garchcaster(
            garch=argarch211_res,
            iters=n,
            periods=days,
        )
        print (cast.order)
        assert cast.order == (2,0,1,1)
        assert cast.mean.m == 2
        assert cast.mean.n == 0
        assert cast.vol.p == 1
        assert cast.vol.q == 1
        assert cast.maxlag == 2
        assert cast.iters == n
        assert cast.periods == days
        assert cast.dist is None
        assert cast.y_hist.size == 2
        assert np.isclose(cast.y_hist[0], 0.7912614766410968, rtol=.00001).all()
        assert cast.vols_hist.size == 2
        assert np.isclose(cast.vols_hist[0], 3.489919888763999, rtol=.00001).all()
        assert cast.resids_hist.size == 2
        assert np.isclose(cast.resids_hist[0], 22.538983051517928, rtol=.00001).all()

    def test_order_warning(self, garch_cmean, n, days):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cast = ph.Garchcaster(
                garch=garch_cmean,
                iters=n,
                periods=days,
                order=(2,0,1,1)
            )
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert 'OVERRIDDEN' in str(w[-1].message)

    # def test_cmean_forecast(self, garch_cmean, n, days):
    #     cast = ph.Garchcaster(
    #         garch=garch_cmean,
    #         iters=n,
    #         periods=days,
    #     )
    #     fore = cast.forecast()
