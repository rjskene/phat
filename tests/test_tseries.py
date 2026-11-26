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


def test_submodules_are_reachable_as_attributes():
    # 0.0.x imported every submodule eagerly, so `ph.bootstrap.numpy.dbl_bs` worked
    code = ("import phat as ph; "
            "print(ph.bootstrap.numpy.dbl_bs.__name__, ph.tseries.Garchcaster.__name__, "
            "ph.fit.fit_phat.__name__)")
    assert _loaded_in_fresh_interpreter(code) == 'dbl_bs Garchcaster fit_phat'


def test_missing_optional_dependency_names_the_extra(monkeypatch):
    import sys
    for mod in [m for m in sys.modules if m.startswith('phat.learn')]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setitem(sys.modules, 'tensorflow', None)
    with pytest.raises(ImportError, match=r'phat-tails\[learn\]'):
        ph.PhatNet  # noqa: B018
    assert not hasattr(ph, 'no_such_name')


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
            n_paths=4, horizon=5,
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
        # values are percentage returns; gross_returns is the price relative
        assert np.allclose(a.gross_returns, 1 + a.values / 100)

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
        cast = Garchcaster(garch=garch_res, n_paths=50, horizon=20)
        assert cast.order == (0, 0, 1, 1)
        assert cast.maxlag == 1
        assert np.isclose(cast.mean.constant, garch_res.params['mu'])
        assert np.isclose(cast.vol.constant, garch_res.params['omega'])
        assert np.isclose(cast.vol.arch_params[0], garch_res.params['alpha[1]'])
        assert np.isclose(cast.vol.garch_params[0], garch_res.params['beta[1]'])
        assert cast.y_hist.size == cast.vols_hist.size == cast.resids_hist.size == 1
        assert np.isclose(cast.y_hist[0], garch_res.model.y.iloc[-1])
        assert np.isclose(cast.resids_hist[0], garch_res.resid.iloc[-1])
        assert np.isclose(cast.vols_hist[0], garch_res.conditional_volatility.iloc[-1])

    def test_forecast_smoke(self, garch_res):
        dist = ph.Phat(0, 1, 0.3, 0.3)
        cast = Garchcaster(garch=garch_res, n_paths=200, horizon=30, dist=dist)
        res = cast.forecast(seed=0)
        assert res.values.shape == (200, 30)
        assert np.isfinite(res.values).all()
        # conditional volatility stays on the scale of the data (daily % returns)
        assert 0.5 < np.sqrt(res.vars).mean() < 20

    def test_order_warning(self, garch_res):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Garchcaster(garch=garch_res, n_paths=5, horizon=5, order=(2, 0, 1, 1))
        assert len(w) == 1 and issubclass(w[-1].category, UserWarning)
        assert 'OVERRIDDEN' in str(w[-1].message)

    def test_rejects_history_arrays_with_arch_result(self, garch_res):
        with pytest.raises(ValueError):
            Garchcaster(garch=garch_res, y=[1.0])


def _reference_paths(c, ar, ma, omega, alpha, beta, y, vols, resids, innovs):
    """Plain-Python ARMA-GARCH recursion; every history runs oldest to newest."""
    out_y, out_var = np.zeros(innovs.shape), np.zeros(innovs.shape)
    for i in range(innovs.shape[0]):
        ys, vs, es = list(y), list(vols), list(resids)
        for t in range(innovs.shape[1]):
            var = omega + sum(a * es[-1 - k] ** 2 for k, a in enumerate(alpha))
            var += sum(b * vs[-1 - k] ** 2 for k, b in enumerate(beta))
            e = np.sqrt(var) * innovs[i, t]
            mean = c + sum(p * ys[-1 - k] for k, p in enumerate(ar))
            mean += sum(q * es[-1 - k] for k, q in enumerate(ma))
            ys.append(mean + e)
            vs.append(np.sqrt(var))
            es.append(e)
            out_y[i, t], out_var[i, t] = mean + e, var
    return out_y, out_var


class TestRecursionWithSeveralLags:
    """Lag k of a history is its k-th value from the end; coefficient k multiplies lag k."""
    y = np.array([0.9, -0.4, 0.25])
    vols = np.array([1.3, 0.8, 1.1])
    resids = np.array([0.7, -0.5, 0.3])
    innovs = np.random.default_rng(3).standard_normal((3, 8))

    def test_arma22_garch11(self):
        c, ar, ma, vol = 0.02, [0.3, -0.2], [0.15, 0.4], [0.05, 0.1, 0.85]
        cast = Garchcaster(
            arma=[c, *ar, *ma], garch=vol, order=(2, 2, 1, 1),
            y=self.y, vols=self.vols, resids=self.resids, n_paths=3, horizon=8,
        )
        res = cast.forecast(innovs=self.innovs)
        want_y, want_var = _reference_paths(
            c, ar, ma, 0.05, [0.1], [0.85], self.y, self.vols, self.resids, self.innovs)
        assert np.allclose(res.values, want_y, rtol=0, atol=1e-12)
        assert np.allclose(res.vars, want_var, rtol=0, atol=1e-12)

    def test_ar3_garch22(self):
        c, ar = -0.01, [0.2, 0.1, -0.3]
        cast = Garchcaster(
            arma=[c, *ar], garch=[0.04, 0.07, 0.03, 0.5, 0.3], order=(3, 0, 2, 2),
            y=self.y, vols=self.vols, resids=self.resids, n_paths=3, horizon=8,
        )
        res = cast.forecast(innovs=self.innovs)
        want_y, want_var = _reference_paths(
            c, ar, [], 0.04, [0.07, 0.03], [0.5, 0.3], self.y, self.vols, self.resids, self.innovs)
        assert np.allclose(res.values, want_y, rtol=0, atol=1e-12)
        assert np.allclose(res.vars, want_var, rtol=0, atol=1e-12)

    def test_use_backcast_is_deprecated_and_has_no_effect(self):
        kwargs = dict(arma=[0.05], garch=[0.1, 0.1, 0.8], y=self.y, vols=self.vols, resids=self.resids,
                      n_paths=3, horizon=8)
        plain = Garchcaster(**kwargs).forecast(innovs=self.innovs)
        with pytest.warns(DeprecationWarning, match='use_backcast'):
            cast = Garchcaster(use_backcast=True, **kwargs)
        assert cast.resids_hist.tolist() == [0.3]
        assert np.array_equal(cast.forecast(innovs=self.innovs).values, plain.values)


@pytest.mark.skipif(not HAS_ARCH, reason='needs the garch extra (arch, pandas)')
class TestPathsMatchArch:
    """On identical innovations a forecast reproduces ``arch``'s own simulation."""
    innovs = np.random.default_rng(11).standard_normal((4, 12))

    @staticmethod
    def _fit(mean, p, q, o=0):
        import pandas as pd
        from arch.univariate import ARX, GARCH, ConstantMean, ZeroMean
        rets = pd.read_csv(RETURNS, index_col='Date')['YELP']
        if mean == 'constant':
            model = ConstantMean(rets)
        elif mean == 'zero':
            model = ZeroMean(rets)
        elif mean == 'ar2 without constant':
            model = ARX(rets, lags=2, constant=False)
        else:
            model = ARX(rets, lags=mean)
        model.volatility = GARCH(p=p, o=o, q=q)
        return model.fit(disp='off')

    def _arch_paths(self, res):
        sim = res.forecast(
            horizon=self.innovs.shape[1], simulations=self.innovs.shape[0],
            method='simulation', rng=lambda size: self.innovs.reshape(size), reindex=False,
        ).simulations
        # `variances` is the variance of the series, which an AR mean accumulates over the
        # horizon; `residual_variances` is the conditional variance of the residual
        return sim.values[0], sim.residual_variances[0]

    @pytest.mark.parametrize('mean, p, q', [
        ('constant', 1, 1),
        ('constant', 2, 2),
        ('constant', 2, 1),
        ('constant', 1, 2),
        ('zero', 1, 1),
        ([1, 2], 1, 1),
        ([1, 2, 3], 2, 1),
        ('ar2 without constant', 1, 1),
    ])
    def test_forecast_matches_arch(self, mean, p, q):
        res = self._fit(mean, p, q)
        want_y, want_var = self._arch_paths(res)
        got = Garchcaster(garch=res, n_paths=4, horizon=12).forecast(innovs=self.innovs)
        assert np.allclose(got.values, want_y, rtol=0, atol=1e-9)
        assert np.allclose(got.vars, want_var, rtol=0, atol=1e-9)

    def test_asymmetric_garch_is_rejected(self):
        res = self._fit('constant', 1, 1, o=1)
        with pytest.raises(NotImplementedError, match='asymmetric'):
            Garchcaster(garch=res)

    def test_unsupported_mean_models_are_rejected(self):
        import pandas as pd
        from arch.univariate import ARX, HARX
        rets = pd.read_csv(RETURNS, index_col='Date')['YELP']
        for model in (ARX(rets, lags=[1, 3]), HARX(rets, lags=[1, 5])):
            with pytest.raises(NotImplementedError, match='consecutive lags'):
                Garchcaster(garch=model.fit(disp='off'))

    def test_ar_history_is_the_observed_series(self):
        res = self._fit([1, 2], 1, 1)
        cast = Garchcaster(garch=res)
        assert np.allclose(cast.y_hist, np.asarray(res.model.y)[-2:])
        assert np.allclose(cast.resids_hist, np.asarray(res.resid)[-2:])


@pytest.mark.skipif(not HAS_ARCH, reason='needs the garch extra (arch, pandas)')
class TestPmdarimaModel:
    """A fitted ``pmdarima.ARIMA`` is read by duck typing; a stand-in is enough here."""
    series = np.array([0.4, -1.1, 0.6, 0.2, -0.3])

    class FakeArima:
        def __init__(self, order, params, series, with_intercept=True, seasonal_order=(0, 0, 0, 0)):
            self.order, self._params = order, np.asarray(params, dtype=float)
            self.with_intercept, self.seasonal_order = with_intercept, seasonal_order
            model = type('Model', (), {'endog': np.asarray(series, dtype=float).reshape(-1, 1)})()
            self.arima_res_ = type('Res', (), {'model': model})()

        def params(self):
            return self._params

    @pytest.fixture(scope='class')
    @staticmethod
    def garch_res():
        return TestPathsMatchArch._fit('constant', 1, 1)

    def test_all_coefficients_are_kept(self, garch_res):
        # constant, three AR, two MA, then the innovation variance
        arma = self.FakeArima((3, 0, 2), [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 9.9], self.series)
        cast = Garchcaster(arma=arma, garch=garch_res)
        assert cast.order == (3, 2, 1, 1)
        assert cast.mean.constant == 0.01
        assert cast.mean.ar_params.tolist() == [0.1, 0.2, 0.3]
        assert cast.mean.ma_params.tolist() == [0.4, 0.5]

    def test_model_without_intercept(self, garch_res):
        arma = self.FakeArima((1, 0, 1), [0.1, 0.4, 9.9], self.series, with_intercept=False)
        cast = Garchcaster(arma=arma, garch=garch_res)
        assert cast.mean.constant == 0.0
        assert cast.mean.ar_params.tolist() == [0.1]
        assert cast.mean.ma_params.tolist() == [0.4]

    def test_ar_history_is_the_series_the_arma_was_fitted_to(self, garch_res):
        arma = self.FakeArima((2, 0, 2), [0.01, 0.1, 0.2, 0.3, 0.4, 9.9], self.series)
        cast = Garchcaster(arma=arma, garch=garch_res)
        assert cast.y_hist.tolist() == [0.2, -0.3]
        assert np.allclose(cast.resids_hist, np.asarray(garch_res.resid)[-2:])

    def test_differenced_or_seasonal_models_are_rejected(self, garch_res):
        with pytest.raises(NotImplementedError, match='differenc'):
            Garchcaster(arma=self.FakeArima((1, 1, 1), [0.0, 0.1, 0.4, 9.9], self.series), garch=garch_res)
        seasonal = self.FakeArima((1, 0, 1), [0.0, 0.1, 0.4, 9.9], self.series, seasonal_order=(1, 0, 0, 5))
        with pytest.raises(NotImplementedError, match='seasonal'):
            Garchcaster(arma=seasonal, garch=garch_res)

    def test_garch_must_be_fitted_to_the_end_of_the_arma_residuals(self, garch_res):
        # the residual and volatility histories come from `garch`, the series from `arma`;
        # they have to end on the same observation
        arma = self.FakeArima((2, 0, 2), [0.01, 0.1, 0.2, 0.3, 0.4, 9.9], self.series)
        tail = np.asarray(garch_res.model.y, dtype=float)[-5:]
        arma.resid = lambda: tail
        Garchcaster(arma=arma, garch=garch_res)                     # aligned: accepted
        arma.resid = lambda: np.append(tail, [0.7, -0.2])           # two later observations
        with pytest.raises(ValueError, match='same observation'):
            Garchcaster(arma=arma, garch=garch_res)
