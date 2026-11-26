API
===

Everything below is importable from the top-level package, for example
``phat.Phat`` and ``phat.Garchcaster``. Only NumPy and SciPy are required for
the distribution classes and the fit; see the introduction for the optional
extras.

Distribution
------------

.. autoclass:: phat.dists.Phat
   :members:
   :member-order: bysource

.. autoclass:: phat.dists.CarBenRight
   :members:
   :inherited-members:
   :member-order: bysource

.. autoclass:: phat.dists.CarBenLeft
   :members: pdf, cdf, sf, ppf, isf, mean, var

.. autoclass:: phat.dists.CarBenHybrid

.. autoclass:: phat.dists.PhatStack
   :members: rvs

Estimation
----------

.. autofunction:: phat.fit.fit_phat

.. autoclass:: phat.fit.PhatFitResult
   :members:

.. autofunction:: phat.bootstrap.numpy.two_tailed_hill_double_bootstrap

.. autofunction:: phat.bootstrap.numpy.seed_bootstrap

Forecasting
-----------

.. autoclass:: phat.tseries.Garchcaster
   :members: dist, forecast

.. autoclass:: phat.tseries.GarchcastResults
   :members: plot, sim
