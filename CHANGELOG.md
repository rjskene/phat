# Changelog

All notable changes to `phat-tails` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Every change in
behaviour is listed, because numbers printed in the old README and notebooks
change.

## [Unreleased]

### Fixed
- `CarBenHybrid(xi=...)` by keyword raised `IndexError` because the
  positional check `len(args) >= 0` was always true; a negative keyword `xi`
  also never flipped sign. Keyword and positional construction now agree, and
  integer / NumPy scalars are accepted for `xi`. The sign-of-`xi` convention
  itself is unchanged (owner decision pending).

### Changed
- `import phat` no longer imports TensorFlow, `arch`, `pmdarima` or Numba.
  `phat.bootstrap`, `phat.learn` and `phat.tseries` load on first use of
  `two_tailed_hill_double_bootstrap`, `PhatNet`, `Garchcaster`, etc.
- `phat.utils` no longer imports pandas or matplotlib at module level.
  `arrayarize` accepts any object with a `to_numpy` method.

### Removed
- `phat/thold.py`, which installed the R package `POT` at import time and was
  listed in `.gitignore` while still tracked. It was never exported by the
  package.
