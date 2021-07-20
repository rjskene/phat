# Pareto Hybrids w Asymmetric Tails #

The **Phat** distribution is a two-tailed, fully-continuous, well-defined asymmetric power law probability distribution.

The package makes available several methods to fit a given time-series dataset to the parameters of the Phat distribution and produce a forecast with the results.

## Installation ##

Installation available via `pip`

```console
$ pip install phat-tails
```

### Dependencies ###

+ Python verisons: 3.9
+ Scipy 1.7.*
+ Scikit-learn 0.24.*
+ Tensorflow 2.5.*
+ Numba 0.53.*

Also see requirements and compatibility specifications for [Tensorflow](https://www.tensorflow.org/install) and [Numba](https://numba.readthedocs.io/en/stable/user/installing.html)

### Suggested ###
+ [arch](https://arch.readthedocs.io/en/latest/): *the* python package for fitting and forecasting GARCH models
+ [pmdarima](http://alkaline-ml.com/pmdarima/): recommend for fitting ARMA models (`arch` currently does not support MA)
+ [statsmodels](https://www.statsmodels.org/): wrapped by both `arch` and `pmdarima`

### Also Check Out ###

+ [tail-estimation](https://github.com/ivanvoitalov/tail-estimation)
    + built as part of [Ivan Voitalov et al (2019)](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.1.033034) on tail index estimation techniques for power law phenomenon in scale-free networks
    + code from this package is utilized in the `two_tailed_hill_double_bootstrap` function

## The Issue with Fat Tails ##

Many phenomena are understood to exhibit fat tails: insurance losses, wealth distribution, [rainfall](https://hess.copernicus.org/articles/17/851/2013/hess-17-851-2013.pdf), etc. These are one-tailed phenomenom (usually bounded by zero) for which many potential distributions are applicable: Weibull, Levy, Frechet, Paretos I-IV, the generalized Pareto, the Extreme Value distribution etc.

For two-tailed phenomenon, such as financial asset returns, there are only two and decidedly imperfect candidates:

+ Levy-Stable Distribion 
    + the Levy-Stable is bounded in the range $\alpha \in (0, 2]$ with $\alpha = 2$ being the Gaussian distribution. Thus, the Levy-Stable *only* exhibits fat tails with tail index $\alpha < 2$
    + Unfortunately, equity returns in particular are known to have both a [second moment](https://fan.princeton.edu/fan/FinEcon/chap1.pdf) AND [fat tails](https://papers.tinbergen.nl/98017.pdf), meaning $\alpha > 2$, which the Levy-Stable does not support.
+ Student's T
    + the Student's T is the most popular distribution for modelling asset returns as it does exhibit fat tails and it is power law-*like*.
    + unfortunately, the Student's T only *tends* toward a power law in the extreme tails and so can still heavily underestimate unlikely events.
    + also, the Student's T is symmetric and cannot accomodate different tail indices in either tail. Nor can the skewed Student's T, which is asymmetric, but accepts only a single tail index.

*we should note that recently an asymmetric Student's T has* [been proposed](https://www.sciencedirect.com/science/article/abs/pii/S0304407610000266) *to address this.*

## the Phat Distribution ##

The Phat Distribution is a mixture model of two Pareto hybrid distributions, as described in [2009 by Julie Carreau and Yoshua Bengio](https://www.researchgate.net/publication/226293435_A_hybrid_Pareto_model_for_asymmetric_fat-tailed_data_The_univariate_case) (and dubbed by us the "CarBen" distribution). The CarBen, in turn, is a piece-wise combination of a single Gaussian distribution and a generalized Pareto distribution fused together at the Pareto location, $a$.

The result is a distribution with Gaussian-body and distinct Pareto power laws in either tail. The distribution requires only 4 parameters:

+ $\mu, \sigma$ in the Gaussian body
+ $\xi_{\text{left}}, \xi_{\text{right}}$, being the inverse tail index for either Paretian tail.

Below, we show a Phat distribution with a standard normal body and symmetric Paretian tails with $\xi = .2$ (corresponding to $\alpha = 5$), highlighting the distributions different sections.

```python
import numpy as np
import scipy.stats as scist
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'whitegrid')

from phat import Phat

shape, mean, sig = 1/5, 0, 1
x = np.linspace(-5+mean, 5+mean, 1000)
phat_dist = Phat(mean, sig, shape, shape)
```

![png](imgs/output_6_0.png)

The Paretian tails are parameterized independently and so allow for asymmetry. Below we show two Phat distributions, one with symmetric tail index of $\alpha=2$ and the other with asymmetric tail indices, $\alpha_{\text{left}}=2$ and $\alpha_{\text{right}}=20$.

![png](imgs/output_8_0.png)

The left tails are identical and, as would be expected, the distribution with the greater tail index has a slightly lower probability in the body and a slightly higher probability out in the tails ... that slightly higher probability leading to dramatically different effects, as we know.

## Demo ##

Users are, of course, welcome to use the Phat distribution in any way they see fit. Below we  show simple process for fitting and projecting a financial time series using `phat`; this example will utilize end-of-day daily prices of Coca-Cola, for which there is data back to 1962.

the Fit:

+ download the daily prices of Coca-Cola (ticker: KO). Find the daily returns in percentage terms (i.e. x 100).
+ use the `arch` package to fit a GARCH(1,1) model to the daily returns
+ use the Hill double bootstrap method to estimate the tail index of both tails of the standardized residuals of the AR-GARCH fit.
+ use `phat` custom data class, `DataSplit`, to split the data into training, testing, and validation subsets. *Be careful to scale by 1/10.*
+ use `PhatNet` and `phat`'s custom loss function `PhatLoss` to fit the remaining parameters.
+ use `Garchcaster` to produce 1,000 simulations of a one-year forecast via the same AR-GARCH model.

```pytnon
import yfinance as yf
import arch
import phat


ko = yf.download('KO')
ko_ret = ko.Close.pct_change().dropna()*100
ko_ret = ko_ret[-252*10:]

res = arch.arch_model(ko_ret, mean='Constant', vol='Garch', p=1, q=1).fit(disp='off')
xi_left, xi_right = phat.two_tailed_hill_double_bootstrap(res.std_resid)

data = phat.DataSplit(res.std_resid[2:]/10)
pnet = phat.PhatNet(neurons=1)
pnet.compile(loss=phat.PhatLoss(xi_left,xi_right), optimizer='adam')
history = pnet.fit(data.train, validation_data=data.test, epochs=100, verbose=0)
```

Below we compare the fit of the Phat distribution to that of the Guassian and the Student's T (the paramters found in the `PhatNet` fit can be found via `predicted_params`).  Note the Student's T fits to $v=4.65$, which is equivalent to $\xi = 0.22$, which is a thinner tail than found through the Hill Double bootstrap, particularly for the left tail.

![png](imgs/output_18_0.png)

Note that the Phat distribution is a better fit to the peak of the distribution while both the normal distribution and Student's T are better fits in the shoulders. The devil, of course, is in the de*tails*.

![png](imgs/output_20_0.png)

Out in the left and right tails we see the Phat distribution does a far-better job capturing hte extreme events that have occured in the past 10 years. See the documentation for why this seemingly minor improvement has such a big impact in the tails. 

We can then feed this distribution, along with the results from the AR-GARCH fit, into the `Garchcaster`.

```python
n = 10000
days = 252*10

mu, sig, l, r = pnet.predicted_params().values
phatdist = Phat(mu*10, sig*10, l, r)
fore = phat.Garchcaster(
    res.std_resid[2:],
    res.conditional_volatility[2:],
    res.resid[2:],
    None,
    res.params,
    iters=n,
    periods=days,
    order=(0,0,1,1),
    dist=phatdist
).forecast()
```

We can visual inspect the forecast via several available plot types.

```python
fore.plot('var')
```

![png](imgs/output_23_0.png)

```python
fore.plot('price', p=ko.Close[-1], n=4)
```

![png](imgs/output_24_0.png)

![png](imgs/output_24_1.png)

![png](imgs/output_24_2.png)

![png](imgs/output_24_3.png)

Customizations can be added as well. Below we see a histogram of the last period prices for each of the 10,000 simulations.

```python
ax, P = fore.plot('end_price', p=ko.Close[-1], ec='C0')
lnorm_fit = scist.lognorm.fit(P[:, -1])
ax.plot(np.linspace(0, 350), scist.lognorm(*lnorm_fit).pdf(np.linspace(0, 350)))
```

![png](imgs/output_25_0.png)
