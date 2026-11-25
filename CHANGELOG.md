# Changelog

All notable changes to `phat-tails` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Every change in
behaviour is listed, because numbers printed in the old README and notebooks
change.

## [Unreleased]

### Fixed
- `Garchcaster.dist` raised nothing when no distribution was set (the
  `NotImplementedError` was constructed but not raised) and
  `Garchcaster` kept `resids[:-maxlag:]` (everything but the last `maxlag`
  values) as its residual history when `use_backcast=False`; it now keeps the
  last `maxlag` residuals.
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
  integer / NumPy scalars are accepted for `xi`. The sign-of-`xi` convention
  itself is unchanged (owner decision pending).

### Added
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

### Changed
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
  pandas or matplotlib: the JIT decorators fall back to plain Python, the
  `arch` `isinstance` checks are skipped, the progress bar becomes a plain
  `range`, and matplotlib is imported inside the plotting methods.
- `Garchcaster` with no innovation distribution draws Gaussian innovations with
  `numpy.random.default_rng(seed)` instead of a Numba loop that ignored `seed`.
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
- `statsmodels` is imported only when `Phat.fit` is called.
- `PhatFit.nloglikeobs` returns `inf` for infeasible parameters instead of
  raising.
- `import phat` no longer imports TensorFlow, `arch`, `pmdarima` or Numba.
  `phat.bootstrap`, `phat.learn` and `phat.tseries` load on first use of
  `two_tailed_hill_double_bootstrap`, `PhatNet`, `Garchcaster`, etc.
- `phat.utils` no longer imports pandas or matplotlib at module level.
  `arrayarize` accepts any object with a `to_numpy` method.

### Removed
- `phat/thold.py`, which installed the R package `POT` at import time and was
  listed in `.gitignore` while still tracked. It was never exported by the
  package.
