"""Garchcaster: order inference, array constructor, arch integration and a forecast smoke test."""
import importlib.util
import os
import pathlib
import warnings

import numpy as np
import pytest

import phat as ph
from phat.tseries import MEAN, PROCESSMIXIN, VOL, Garchcaster, GarchcastResults

HAS_ARCH = importlib.util.find_spec('arch') is not None and importlib.util.find_spec('pandas') is not None
RETURNS = pathlib.Path(__file__).parent / 'yelp_returns.csv'


HEAVY = "{'tensorflow','arch','pmdarima','numba','matplotlib','pandas','statsmodels','tqdm','sklearn'}"


def _loaded_in_fresh_interpreter(code: str) -> str:
    import subprocess
    import sys
    env = {**os.environ, 'PYTHONPATH': str(pathlib.Path(ph.__file__).resolve().parent.parent)}
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, env=env)
    return out.stdout.strip()


def test_import_phat_imports_numpy_and_scipy_only():
    code = ("import sys, phat; phat.Phat(0, 1, .2, .2).cdf(1.0); "
            f"print(sorted({{m.split('.')[0] for m in sys.modules}} & {HEAVY}))")
    assert _loaded_in_fresh_interpreter(code) == '[]'


def test_garchcaster_and_bootstrap_do_not_import_tensorflow_arch_or_pandas():
    # numba and tqdm are accelerators that load when installed; arch, pandas and the
    # TensorFlow stack never load until a fitted arch model or phat.learn is used.
    code = ("import sys, phat; from phat import Garchcaster, two_tailed_hill_double_bootstrap; "
            f"print(sorted({{m.split('.')[0] for m in sys.modules}} & {HEAVY} - {{'numba', 'tqdm'}}))")
    assert _loaded_in_fresh_interpreter(code) == '[]'


class TestOrderInference:
    def test_infer_order(self):
        om = PROCESSMIXIN()
        assert om._infer_order(np.array([1., 2., 3., 4.])) == (2, 2)
        assert om._infer_order(np.array([1., 2., 3.])) == (2, 1)
        assert om._infer_order(np.array([1., 2.])) == (1, 1)
        assert om._infer_order(np.array([1.])) == (1, 0)
        assert om._infer_order(np.array([])) == (0, 0)
        assert om._infer_order(None) == (0, 0)

    def test_constant_mean_container(self):
        mean = MEAN(params=np.array([0.14]))
        assert np.isclose(mean.constant, 0.14)
        assert (mean.m, mean.n, mean.order, mean.max_order) == (0, 0, (0, 0), 0)
        assert not mean.ar_params.any() and not mean.ma_params.any()

    def test_ar2_mean_container(self):
        mean = MEAN(params=np.array([0.15, -0.05, 0.01]), order=(2, 0))
        assert np.isclose(mean.constant, 0.15)
        assert np.allclose(mean.ar_params, [-0.05, 0.01])
        assert (mean.m, mean.n, mean.max_order) == (2, 0, 2)
        assert not mean.ma_params.any()

    def test_garch11_vol_container(self):
        vol = VOL(params=np.array([4.15, 0.14, 0.57]))
        assert np.isclose(vol.constant, 4.15)
        assert (vol.p, vol.q, vol.order, vol.max_order) == (1, 1, (1, 1), 1)
        assert np.isclose(vol.arch_params[0], 0.14)
        assert np.isclose(vol.garch_params[0], 0.57)


class TestArrayConstructor:
    @pytest.fixture
    def cast(self):
        return Garchcaster(
            garch=[0.1, 0.1, 0.8],          # omega, alpha, beta
            arma=[0.05],                    # constant mean
            y=np.array([0.5, -0.3]),
            vols=np.array([1.1, 1.2]),
            resids=np.array([0.4, -0.6]),
            iters=4, periods=5, use_backcast=False,
        )

    def test_order_and_history(self, cast):
        assert cast.order == (0, 0, 1, 1)
        assert cast.maxlag == 1
        assert cast.y_hist.tolist() == [-0.3]
        assert cast.vols_hist.tolist() == [1.2]
        # last `maxlag` residuals (the slice was resids[:-maxlag:] before the fix)
        assert cast.resids_hist.tolist() == [-0.6]

    def test_dist_is_none_for_gaussian_innovations(self, cast):
        assert cast.dist is None

    def test_gaussian_forecast_is_seeded(self, cast):
        a = cast.forecast(seed=3)
        b = cast.forecast(seed=3)
        c = cast.forecast(seed=4)
        assert isinstance(a, GarchcastResults)
        assert a.values.shape == (4, 5)
        assert np.array_equal(a.values, b.values)
        assert not np.array_equal(a.values, c.values)
        assert (a.vars > 0).all()
        # values are percentage returns; rets is the gross return
        assert np.allclose(a.rets, 1 + a.values / 100)

    def test_phat_innovations(self, cast):
        dist = ph.Phat(0, 1, 0.2, 0.25)
        res = cast.forecast(dist=dist, seed=1)
        assert res.values.shape == (4, 5)
        assert np.isfinite(res.values).all()

    def test_phat_innovations_use_the_exact_unit_variance_sampler(self, cast):
        dist = ph.Phat(0, 1, 0.2, 0.25)
        rng = np.random.default_rng(7)
        innovs = cast._sample_innovs(400, 50, dist=dist, random_state=rng)
        expected = dist.std_rvs(size=(400, 50), random_state=np.random.default_rng(7))
        assert np.array_equal(innovs, expected)
        assert abs(innovs.std() - 1) < 0.05
        assert abs(innovs.mean() - dist.mean() / dist.std()) < 0.05

    def test_random_state_accepts_generator_and_matches_seed(self, cast):
        a = cast.forecast(random_state=np.random.default_rng(11))
        b = cast.forecast(seed=11)
        c = cast.forecast(random_state=11)
        assert np.array_equal(a.values, b.values)
        assert np.array_equal(a.values, c.values)

    def test_innovs_shape_is_checked(self, cast):
        with pytest.raises(ValueError):
            cast.forecast(innovs=np.zeros((3, 5)))

    def test_scipy_frozen_innovations(self, cast):
        import scipy.stats
        res = cast.forecast(dist=scipy.stats.t(5), seed=2)
        assert res.values.shape == (4, 5)
        assert np.isfinite(res.values).all()

    def test_make_sim_and_sim_property(self, cast):
        res = cast.forecast(seed=1)
        with pytest.raises(ValueError):
            _ = res.sim
        sim = res._make_sim(p=100.0)
        assert res.sim is sim

    @pytest.mark.skipif(importlib.util.find_spec('matplotlib') is None, reason='needs the plot extra')
    @pytest.mark.parametrize('kind', ['vol', 'var', 'price', 'end_price'])
    def test_plot_kinds(self, cast, kind):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        res = cast.forecast(seed=1)
        out = res.plot(kind=kind, p=100.0, n=2)
        assert out is not None
        plt.close('all')

    def test_plot_rejects_unknown_kind(self, cast):
        pytest.importorskip('matplotlib')
        res = cast.forecast(seed=1)
        with pytest.raises(ValueError):
            res.plot(kind='nope')

    def test_garch_variance_recursion(self, cast):
        # one step: var = omega + alpha*resid^2 + beta*vol^2, resid = sqrt(var)*innov, y = c + resid
        innovs = np.full((4, 5), 0.5)
        res = cast.forecast(innovs=innovs)
        var1 = 0.1 + 0.1 * (-0.6) ** 2 + 0.8 * 1.2 ** 2
        assert np.allclose(res.vars[:, 0], var1)
        assert np.allclose(res.resids[:, 0], np.sqrt(var1) * 0.5)
        assert np.allclose(res.values[:, 0], 0.05 + np.sqrt(var1) * 0.5)
        var2 = 0.1 + 0.1 * res.resids[0, 0] ** 2 + 0.8 * var1
        assert np.allclose(res.vars[:, 1], var2)


@pytest.mark.skipif(not HAS_ARCH, reason='needs the garch extra (arch, pandas)')
class TestArchIntegration:
    @pytest.fixture(scope='class')
    @staticmethod
    def garch_res():
        import pandas as pd
        from arch import arch_model
        rets = pd.read_csv(RETURNS, index_col='Date')['YELP']
        return arch_model(rets, mean='Constant', vol='GARCH', p=1, q=1).fit(disp='off')

    def test_constant_mean_garch11(self, garch_res):
        cast = Garchcaster(garch=garch_res, iters=50, periods=20)
        assert cast.order == (0, 0, 1, 1)
        assert cast.maxlag == 1
        assert np.isclose(cast.mean.constant, garch_res.params['mu'])
        assert np.isclose(cast.vol.constant, garch_res.params['omega'])
        assert np.isclose(cast.vol.arch_params[0], garch_res.params['alpha[1]'])
        assert np.isclose(cast.vol.garch_params[0], garch_res.params['beta[1]'])
        assert cast.y_hist.size == cast.vols_hist.size == cast.resids_hist.size == 1
        assert np.isclose(cast.y_hist[0], garch_res.std_resid.iloc[-1])
        assert np.isclose(cast.vols_hist[0], garch_res.conditional_volatility.iloc[-1])

    def test_forecast_smoke(self, garch_res):
        dist = ph.Phat(0, 1, 0.3, 0.3)
        cast = Garchcaster(garch=garch_res, iters=200, periods=30, dist=dist)
        res = cast.forecast(seed=0)
        assert res.values.shape == (200, 30)
        assert np.isfinite(res.values).all()
        # conditional volatility stays on the scale of the data (daily % returns)
        assert 0.5 < np.sqrt(res.vars).mean() < 20

    def test_order_warning(self, garch_res):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Garchcaster(garch=garch_res, iters=5, periods=5, order=(2, 0, 1, 1))
        assert len(w) == 1 and issubclass(w[-1].category, UserWarning)
        assert 'OVERRIDDEN' in str(w[-1].message)

    def test_rejects_history_arrays_with_arch_result(self, garch_res):
        with pytest.raises(ValueError):
            Garchcaster(garch=garch_res, y=[1.0])
