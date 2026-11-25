# Changelog

All notable changes to `phat-tails` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Every change in
behaviour is listed, because numbers printed in the old README and notebooks
change.

## [Unreleased]

### Fixed
- `Garchcaster` kept `resids[:-maxlag:]` (everything but the last `maxlag`
  values) as its residual history when `use_backcast=False`; it now keeps the
  last `maxlag` residuals. `Garchcaster.dist` is documented to return `None`
  when no innovation distribution was given (Gaussian innovations); the dead
  `NotImplementedError` expression there is gone.
- `Phat.fit` results name their parameters `mu`, `sig`, `xi_l`, `xi_r`
  (previously `const`, `par0`, ...) and report the right degrees of freedom.
- `Phat.ppf` returned the weighted average of the two component quantile
  functions, which is not the quantile of a mixture. It is now the root of the
  mixture CDF (vectorised, safeguarded Newton with a guaranteed bracket). For
  `Phat(0, 1, 0.2, 0.25)` the 1% quantile moves from -10.01 to -13.86 and the
  99.9% quantile from 23.58 to 35.49.
- `Phat.rvs`, `Phat.std_rvs` and `PhatStack.rvs` sampled through the wrong
  `ppf`. They now draw a component with probability `p` and invert that
  component's CDF; the Kolmogorov-Smirnov distance to the model's own CDF at
  n = 200,000 drops from 0.046 to about 0.002.
- `CarBenRight.mean/var`, `CarBenLeft.mean/var`, `Phat.mean/var/std` omitted the
  body weight `Phi(s)`, passed unstandardised bounds to `truncnorm` (wrong
  whenever `mu != 0` or `sig != 1`) and took the mixture variance as the mean
  of the component variances. `Phat(0, 1, 0.2, 0.25)` now reports mean 0.0778
  and standard deviation 4.905 (previously 0.0796 and 3.793);
  `Phat(2, 2, 0.2, 0.2)` mean 2.000 and sd 9.206 (previously 3.370 and 6.987).
  The mean is `inf`/`-inf`/`nan` when a tail has `xi >= 1` and the variance is
  `inf` when a tail has `xi >= 1/2`.
- `CarBenBase._calc_b_w_a` (component built from the junction `a` instead of
  `sig`) returned the wrong GPD scale. Not in the review brief; found while
  rewriting the components.
- `CarBenHybrid(xi=...)` by keyword raised `IndexError` because the
  positional check `len(args) >= 0` was always true; a negative keyword `xi`
  also never flipped sign. Keyword and positional construction now agree, and
  integer / NumPy scalars are accepted for `xi`.
- `CarBenRight.isf(1.0)` (and so `CarBenLeft.ppf(1.0)`) could return a large
  finite number instead of `-inf` / `inf` through rounding in
  `gamma * p - Phi(s)`; the body branch is now `body.ppf(gamma * (1 - p))`.
- `GarchcastResults.plot(kind='price')` and `plot(kind='end_price')` never
  worked: `p` and `n` were read from the wrong positions (`NameError` when
  omitted) and the `show_chart` flag was dropped by `argsetter`, so the price
  simulation returned two values where three were unpacked. Both now take
  `p=` (starting price) and `n=` (paths) as keywords.

### Added
- Test suite (`pytest`, 215 tests): density integrates to one, CDF monotone with
  the right limits, `ppf(cdf(x)) = x`, sampler Kolmogorov-Smirnov against the
  model CDF, moments and expected shortfall against quadrature, location-scale
  behaviour, C1 continuity at the junctions, moment-existence rules, the
  reference tables from the review, maximum-likelihood parameter recovery, a
  Hill double-bootstrap regression on a committed text fixture
  (`tests/fixtures/student_t3_n1500.csv`), and `Garchcaster` tests (array
  constructor, GARCH recursion by hand, `arch` integration and a forecast smoke
  test). An independent quadrature oracle lives in `tests/oracle.py`.
- GitHub Actions workflow: ruff and mypy; pytest on Linux, macOS (Apple
  Silicon) and Windows for Python 3.11-3.13 with the `dev` extras; and a
  core-only job that installs nothing but NumPy and SciPy.
- `two_tailed_hill_double_bootstrap(..., seed=...)` and
  `phat.bootstrap.seed_bootstrap` make the resampling reproducible (the Numba
  and pure-Python paths use different streams).
- `Phat.sf`, `logsf`, `logcdf`, `logpdf` computed from the GPD and Gaussian
  pieces directly, finite and accurate far into both tails (`sf` was `1 - cdf`).
- `Phat.isf`, `Phat.stats`, `Phat.expected_shortfall(q, tail=...)` (alias
  `es`), accessors `Phat.junctions`, `Phat.body_mass`, `Phat.tail_masses`;
  component accessors `s`, `body_mass`, `tail_mass`, `isf`, `logpdf`,
  `logcdf`, `logsf`, `lower_partial_mean`, `upper_partial_mean`.
- `rvs` on all classes accepts `random_state` (`None`, int,
  `numpy.random.Generator` or `RandomState`); `seed=` is kept as an alias.
- `Phat` validates its parameters: `sig > 0`, `xi_l, xi_r >= 0`, `p` two
  non-negative weights summing to 1.
- `phat.fit.fit_phat` and `phat.fit.PhatFitResult`: maximum-likelihood fit
  with `scipy.optimize` (L-BFGS-B) that needs NumPy and SciPy only. Full
  four-parameter fit, or `mu, sig` with `xi_l, xi_r` fixed (the two-stage
  estimator), optionally with fixed mixture weights `p`. The result carries
  `params`, `names`, `bse` (standard errors from the inverse observed
  information, a finite-difference Hessian at the optimum), `cov`, `llf`,
  `nobs`, `aic`, `bic`, `conf_int()`, `summary()` and `dist` (the fitted
  `Phat`). Standard errors agree with the `statsmodels` wrapper to three
  decimals on a 5,000-point sample and the fit is about four times faster.
- `Garchcaster.forecast` and `_sample_innovs` accept `random_state`
  (`None`, int, `numpy.random.Generator` or `RandomState`) like `rvs`;
  `seed=` stays as an alias. `forecast(innovs=...)` checks the array has shape
  `(iters, periods)`.
- `GarchcastResults.plot(kind='price', random_state=...)` picks the displayed
  paths reproducibly.

### Changed
- **Argument and attribute names (old names deprecated, still working).**
  Positional calls are unaffected: the order of every signature is the same.

  | 0.0.x | now | where |
  |---|---|---|
  | `sig` | `sigma` | `Phat`, `CarBenRight`, `CarBenLeft`, `CarBenHybrid` |
  | `xi_l`, `xi_r` | `xi_left`, `xi_right` | `Phat`, `Phat.fit`, `fit_phat` |
  | `p` | `weights` | `Phat`, `PhatStack`, `Phat.fit` |
  | `rtail=True` / `False` | `tail='right'` / `'left'` | `CarBenHybrid` |
  | `iters`, `periods` | `n_paths`, `horizon` | `Garchcaster`, `Garchcaster.forecast`, `GarchcastResults` |
  | `iters` | `n_runs` | `two_tailed_hill_double_bootstrap` |

  - The old keywords and the old attributes (`dist.sig`, `dist.xi_l`,
    `dist.xi_r`, `dist.p`, `cast.iters`, `cast.periods`, ...) keep working
    for one release and emit a `DeprecationWarning` attributed to the
    calling line. Passing an old and a new name together is a `TypeError`.
  - `Phat.params` keys are `mu, sigma, xi_left, xi_right, a_left, a_right,
    b_left, b_right`; the old keys (`sig`, `xi_l`, `a_l`, ...) still resolve
    with a warning. `Phat.PARAM_NAMES` lists the new names.
  - Fit results name their parameters `mu, sigma, xi_left, xi_right`
    (`PhatFitResult.names`, `params_dict`, `fixed`, and the `statsmodels`
    backend's `exog_names`).
  - `phat.learn` (the TensorFlow module) is unchanged; it has its own names
    and its future is an open decision.
- **Sign convention of `xi` (breaking, with a one-release shim).** `xi` is
  now always the shape of the generalized Pareto tail: positive for a heavy
  tail, zero for an exponential tail, `-1 < xi < 0` for a bounded tail. The
  side is chosen separately, by `CarBenRight` / `CarBenLeft` or by
  `CarBenHybrid(xi, mu, sigma, tail='right'|'left')`.
  - In 0.0.x `CarBenHybrid(-0.25, mu, sig)` meant "left tail with shape
    0.25". That call still returns the same object in this release but emits
    a `FutureWarning`; in the next release it will be a bounded *right* tail.
    Write `CarBenHybrid(0.25, mu, sigma, tail='left')` or
    `CarBenLeft(0.25, mu, sigma)`.
  - With a side given (`tail=`, or the deprecated `rtail=`), a negative
    `xi` is already the bounded tail on that side. `CarBenHybrid(-0.25, mu, sig, rtail=False)` used to raise
    `ValueError` and `CarBenHybrid(-0.25, mu, sig, rtail=True)` used to
    return a left tail with shape 0.25.
  - `Phat(mu, sigma, xi_left, xi_right)` accepts shapes `> -1` (previously
    `>= 0`). Every existing call is unchanged. A component with a negative
    shape has a GPD tail that ends at `a + b/|xi|`; the mixture stays
    supported on the whole line through the other component's Gaussian body.
    All methods are checked against the quadrature oracle for bounded tails.
    `Phat.fit` still searches `0 <= xi <= 10`.
  - `CarBenHybrid` has an explicit signature `(xi, mu, sigma=None, a=None,
    tail='right')` instead of `*args, **kwargs`.
- Packaging moved to a single `pyproject.toml` (`setup.py`, `setup.cfg`,
  `requirements.txt` and `docs/_dev_requirements.txt` removed). Core
  dependencies are `numpy>=1.25` and `scipy>=1.11` only; `requires-python`
  is `>=3.11` (the oldest CPython still supported upstream). Everything else
  is an extra: `fit` (statsmodels), `speed` (numba, tqdm), `garch` (arch,
  pandas), `plot` (matplotlib), `learn` (TensorFlow stack), `docs`, `dev`,
  `all`. `pmdarima` is no longer a dependency; a fitted `pmdarima.ARIMA` is
  still accepted by `Garchcaster` through duck typing.
- Project URL corrected to `https://github.com/rjskene/phat` and the licence
  classifier to MIT, matching `LICENSE`.
- `phat.tseries` and `phat.bootstrap` import without Numba, `arch`, tqdm,
  pandas or matplotlib: the JIT decorators fall back to plain Python, fitted
  `arch` and `pmdarima` models are recognised by duck typing (neither package
  is imported), the progress bar becomes a plain `range`, and matplotlib is
  imported inside the plotting methods.
- Source is formatted to pass `ruff` (E, F, W, I, B, UP) and `mypy` (lenient:
  implicit Optional allowed; `phat.learn` and `phat.bootstrap.tf` are excluded
  pending the owner's decision on PhatNet). `phat.tseries` is fully typed:
  `MEAN`/`VOL` store parameters as float arrays and orders as `(int, int)`
  tuples, `Garchcaster` raises `ValueError` (instead of `UnboundLocalError`)
  when neither parameters nor a fitted model are given.
- `Garchcaster` with no innovation distribution draws Gaussian innovations with
  `numpy.random.default_rng(seed)` instead of a Numba loop that ignored `seed`.
  A `Phat` innovation distribution is sampled with the exact unit-variance
  sampler `Phat.std_rvs(random_state=...)` (the fixed component-then-inverse
  CDF sampler above), so GARCH forecasts with Phat innovations no longer
  inherit the old `ppf` error.
- `Phat.fit(values, xi_l=None, xi_r=None, p=None, *, backend='scipy',
  **kwargs)` returns a `PhatFitResult` by default and no longer needs
  `statsmodels`. `backend='statsmodels'` returns the former
  `GenericLikelihoodModel` results object (needs the `fit` extra). The old
  call `Phat.fit(x, xi_l, xi_r)` still works; `res.params` is now a NumPy
  array rather than a pandas Series, so index by position or use
  `res.params_dict`.
- `PhatFit` moved to the new module `phat.fit`, which is the only module that
  imports `statsmodels`; `phat.dists.PhatFit` and `phat.PhatFit` still resolve
  to it lazily.
- `two_tailed_hill_double_bootstrap` accepts any array-like, no longer calls
  `np.seterr` globally (uses `np.errstate`), and no longer uses a mutable
  default for `pbar_kwargs`. Its docstring credits the `tail-estimation`
  code (Voitalov et al., MIT) it was adapted from.
- `.readthedocs.yaml` builds on Python 3.12 with the `docs` extra;
  `docs/conf.py` reads the version from package metadata and points the
  repository link at `rjskene/phat`.
- Distribution methods return NumPy scalars for scalar input and arrays of the
  input's shape otherwise (SciPy convention). Previously a scalar input gave a
  shape-`(1,)` array, so `dist.cdf(3)[0]` must become `dist.cdf(3)`.
- Random draws use `numpy.random.default_rng`; sequences for a given seed differ
  from 0.0.x.
- `CarBenLeft` is implemented as the reflection of `CarBenRight` about `mu`.
- `Phat.nll` and `Phat.loglike` are per observation and go through `logpdf`.
- `statsmodels` is imported only when `Phat.fit(..., backend='statsmodels')`
  or `phat.fit.PhatFit` is used.
- `PhatFit.nloglikeobs` returns `inf` for infeasible parameters instead of
  raising.
- `import phat` no longer imports TensorFlow, `arch`, `pmdarima` or Numba.
  `phat.bootstrap`, `phat.learn` and `phat.tseries` load on first use of
  `two_tailed_hill_double_bootstrap`, `PhatNet`, `Garchcaster`, etc.
- `phat.utils` no longer imports pandas or matplotlib at module level.
  `arrayarize` accepts any object with a `to_numpy` method.

### Removed
- `tests/test_garchcast.py` and its nine pickled fixtures (fitted `arch` and
  `pmdarima` objects from 2021 that no current library version can unpickle).
  The replacement fits the model from `tests/yelp_returns.csv` at test time.
- `phat/thold.py`, which installed the R package `POT` at import time and was
  listed in `.gitignore` while still tracked. It was never exported by the
  package.
