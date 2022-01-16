"""
Houses all scipy-based distributions.

Separate code required for Tensorflow. Found in .learn.dists

WARNING: Only CarBen and Phat distributions should be used. The remaining
distributions are for illustration purposes in the docs.
"""
from typing import Union, Iterable, Sequence, Callable, List
import numpy as np
import scipy.stats as scist
import scipy.special as spec
from scipy._lib._util import check_random_state
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

from phat.utils import argsetter, arrayarize, sech_squared, stacker, dotweight, XTYPE

class BLineSig:
    """
    Borderline Sigmoid function defined by Taleb in SCOFT https://arxiv.org/abs/2001.10488

    FOR ILLUSTRATION PURPOSES ONLY
    """
    def __init__(self, k:float=1):
        self.k = k

    @argsetter('xk')
    def sf(self, x, k:float=None):
        return .5*(1 - np.tanh(k*x))

    @argsetter('xk')
    def pdf(self, x, k:float=None):
        return .5*k*sech_squared(x,k)
    
    @argsetter('k')
    def var(self, k):
        return np.pi**2 / (12*k**2)
    
    @argsetter('k')
    def m4(self, k):
        return 7*np.pi**4 / (240*k**4)
    
    @argsetter('k')
    def kurt(self, k):
        return self.m4(k) / self.var(k)**2

class LogPareto:
    """
    LogPareto as discussed in SCOFT https://arxiv.org/abs/2001.10488.

    FOR ILLUSTRATION PURPOSES ONLY
    """
    def __init__(self, alpha=2, L=1):
        self.alpha = alpha
        self.L = L
        
    def sf(self, x):
        return (self.L**self.alpha)*(np.log(x)**(-self.alpha))

class DBLFat(scist.rv_continuous):
    """
    Baseclass for group of twin-tailed fat-tailed distributions

    FOR ILLUSTRATION PURPOSES ONLY    
    """
    @property
    def dist(self):
        raise NotImplementedError

    @property
    def qjunc(self):
        return .5

    @argsetter(['x', 'shape'])
    def _pdf(self, x, shape):
        pdf = np.where(
            x>=0, 
            self.dist.pdf(x, shape), 
            self.dist.pdf(-x, shape)
        )
        return pdf / 2
    
    @argsetter(['x', 'shape'])
    def _cdf(self, x, shape):
        cdf = np.where(
            x>=0,
            .5 + (self.dist.cdf(x, shape)/2), 
            
            self.dist.sf(-x, shape)/2
        )
        return cdf
    
    @argsetter(['q', 'shape'])
    def _ppf(self, q, shape):
        if isinstance(q, (float, int)):
            q = np.array([q])
        
        ppf = np.where(
            q>=self.qjunc,
            self.dist.ppf(1/self.qjunc*(q - self.qjunc), shape),
            -self.dist.isf(q*1/self.qjunc, shape)
        )

        return ppf
    
    def _rvs(self, *args, **kwargs):
        rvs = super()._rvs(*args, **kwargs)
        return rvs
    
class DBLGP(DBLFat):
    dist = scist.genpareto
    
class DBLLomax(DBLFat):
    dist = scist.lomax
    
class DBLPareto(DBLFat):
    dist = scist.pareto

############# Main Phat-related Distributions ################
class CarBenBase:
    """
    Base Class of CarBen Hybrid distribution. Generalizes code used in both Left and Right tailed versions.

    CarBen is a one-tailed assymetric hybrid of Gaussian and generalized Pareto distributions. Derivation of 
    the distribution found in Carreau and Bengio (2008):
    https://www.researchgate.net/publication/226293435_A_hybrid_Pareto_model_for_asymmetric_fat-tailed_data_The_univariate_case#pf6

    Takes 3 free parameters:
        > shape (xi) of generalized Pareto (inverse of the tail index)
        > mean (mu), standard deviation (sig) of the Gaussian distribution
    
    Remaining 2 Pareto parameters are found via equalities
        > b, which is the scale, as function of sig and xi (utilizing a Lambert W)
        > a, which is the location, can then be found either via a) scale OR b) via the Lambert W. the latter is used
    
    Optional parameters:
        > instead of `sig`, user can provide `a`, which is the location of the generalized Pareto

    Parameters
    -----------
    xi:     inverse of tail index of generalized Pareto tail
    mu:     mean of the Gaussian body
    sig:    standard deviation of the Gaussian body

    Attributes
    -----------
    body:   scipy gaussian distribution with arguments mean/loc=mu and scale=sig
    tail:   scipy generalized pareto with arguments shape=xi, loc=a, scale=b
    gamma:  the proportionate factor whose inverse ensures pdf of CarBen integrates to 1

    Methods
    --------
        > handle helper and common distribution functions that are agnostic to tail-direction
    """

    def __init__(self, 
        xi:float,
        mu:float,
        sig:float=None, 
        a:float=None
        ) -> None:
        """
        Calcs sig, a, b paramters based on arguments provided

        Assigns body distribution with given arguments. The tail distribution is assigned in child classes.
        """
        self.xi = xi
        self.mu = mu

        provided = (sig is None, a is None)
        if all(provided) or not any(provided):
            raise ValueError('Please provide only one of `sig` or `a`')
        elif sig is not None:
            self.sig = sig
            self._given = 'sig'
            self.b = self._calc_b_w_sig()
            self.a = self._calc_a()
        elif a is not None:
            self.a = a
            self._given = 'a'
            self.b = self._calc_b_w_a()
            self.sig = self._calc_sig()

        self.body = scist.norm(self.mu, self.sig)
        
    @property
    def tail(self) -> str:
        """
        Tail distribution of type scipy.genpareto.

        `_tail` is assigned in CarbenLeft or CarbenRight child class
        """
        if hasattr(self, '_tail'):
            return self._tail
        else:
            raise NotImplementedError

    @property
    def gamma(self) -> float:
        """
        defined in Carreau and Bengio (2008)

        proportionate factor whose inverse scales the CarBen so the density integrates to 1
        """
        val = np.sqrt(self.W_z() / 2)
        return 1 + .5*(1 + spec.erf(val))

    def z(self) -> float:
        """
        in Carreau and Bengio (2008), z defined as:

            (1 + xi)**2 / 2*pi
        """
        num = (1 + self.xi)**2
        denom = 2*np.pi
        return  num / denom        

    def W_z(self) -> float:
        """
        in Carreau and Bengio (2008), W(z) can be calculated directly as:
            
            sig**2 * (1 + xi)**2 / b**2

        b is not a free parameter, so W(z) must be calculated via
        the Lambert function, which is defined only in free parameters.
        """

        val = spec.lambertw(self.z())
        assert (val.imag == 0).all(), val.imag

        return val.real

    def _calc_b_w_sig(self) -> float:
        """
        see Equation 7 in Carreau and Bengio (2008)
        """
        num = self.sig*(1 + self.xi)
        denom = np.sqrt(self.W_z())
        
        return num / denom

    def _calc_b_w_a(self) -> float:
        """
        found by substituion of Equation 7 in Carreau and Bengio (2008)

        sig = (a - mu)/W(z)
        b = ((1+xi)(a - mu)/(W(z)sqrt(W(z)))
        b = a - mu + xi*a - xi*mu
        """
        num = self.a - self.mu*self.xi
        denom = np.sqrt(self.W_z())
        return num / denom

    @argsetter('x')
    def sf(self, x:XTYPE=None) -> np.ndarray:
        """
        Survival function, P(X < x)

        Parameters
        ----------
        x:      float, m x n iterable of floats of random variable samples

        Return
        -------
        float or iterable of floats returning survival probability of each sample
        """
        return 1 - self.cdf(x)
    
    def rvs(self, 
        size:Union[int, Iterable[Sequence[int]]]=None, 
        seed:int=None) -> XTYPE:
        """
        Generates random draws

        1. Draws randomly from [0,1] uniform distribution
        2. Calls quantile function for each draw

        Mimics scipy process:
        https://github.com/scipy/scipy/blob/28e4811d48c99b7c68e41992eda5cff859c1fa2b/scipy/stats/_distn_infrastructure.py#L1032

        Params
        -------
        size:   int or iterable of ints providing dimensions of return array
        seed:   int; initialize with specific random state for replication
        """
        random_state = check_random_state(seed)

        U = random_state.uniform(size=size)
        if size is None:
            U = np.array(U, ndmin=1)
            Y = self.ppf(U)[0]
        else:
            Y = self.ppf(U)
        return Y

class CarBenRight(CarBenBase):
    """
    Right-tailed CarBen distribution.

    Assigns `_tail`, handler methods, and common distribution functions that 
    are specific to the tail direction.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tail = scist.genpareto(self.xi, loc=self.a, scale=self.b)
    
    def _calc_a(self) -> float:
        """
        see Equation 8 in Carreau and Bengio (2008)        
        """
        return self.mu + self.sig*np.sqrt(self.W_z())

    def _calc_sig(self) -> float:
        """
        rearrangement of Equation 8 in Carreau and Bengio (2008)        
        """
        return (self.a - self.mu) / np.sqrt(self.W_z())

    @argsetter('x')
    def pdf(self, x:XTYPE=None) -> np.ndarray:
        """
        Probability density function

        See Carreau and Bengio (2008) for derivation

        Parameters
        ----------
        x:      float, m x n iterable of floats of random variable samples

        Return
        -------
        float or iterable of floats returning likelihood of occurence of each sample
        """        
        pdf = np.where(
            x > self.a,
            self.tail.pdf(x),
            self.body.pdf(x)
        )
        return pdf / self.gamma
    
    @argsetter('x')
    def cdf(self, x:XTYPE=None) -> np.ndarray:
        """
        Cumulative Distribtion Function, P(X<x)

        See Docs for derivation

        Parameters
        ----------
        x:      float, m x n iterable of floats of random variable samples

        Return
        -------
        float or iterable of floats returning cumulative probability of each sample
        """
        cdf = np.where(
            x > self.a,
            self.body.cdf(self.a) + self.tail.cdf(x),
            self.body.cdf(x)
        )
        return cdf / self.gamma 

    @property
    def qjunc(self) -> float:
        """
        Quantile of junction point, a, b/w body and tail distribution
        
        See Docs for derivation
        """
        return 1 - 1 / self.gamma

    def ppf(self, q:XTYPE) -> np.ndarray:
        """
        Quantile function, analogous to `ppf` method in `scipy`.

        See Docs for derivation

        Parameters
        ----------
        q:      float, iterable of floats between [0, 1]

        Return
        -------
        float or iterable of floats returning x values corresponding to the given quantiles
        """
        ppf = np.where(
            q > self.qjunc,
            self.tail.ppf(self.gamma * (q - self.qjunc)),
            self.body.ppf(self.gamma*q)
        )
        return ppf    

    def mean(self) -> float:
        """
        First moment

        See Docs for derivation
        """
        bmu = scist.truncnorm(
            -np.inf, 
            self.a, 
            *self.body.args
        ).mean()
        tmu = self.tail.mean()
        rmu = (bmu + tmu) / self.gamma

        return rmu

    def var(self) -> float:
        """
        Second moment

        See Docs for derivation
        """
        bvar = scist.truncnorm(
            -np.inf, 
            self.a, 
            *self.body.args
        ).var()
        tvar = self.tail.var()
        rvar = (bvar + tvar) / self.gamma
        return rvar

class CarBenLeft(CarBenBase):
    """
    Right-tailed CarBen distribution.

    Assigns `_tail`, handler methods, and common distribution functions that 
    are specific to the tail direction.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # NOTE: `a` is inverted to reflect the distribution appropriately
        self._tail = scist.genpareto(self.xi, loc=-self.a, scale=self.b)

    def _calc_a(self) -> float:
        """
        see Equation 8 in Carreau and Bengio (2008)   

        For the left-tail, the valence of second term is changed to account for 
        reflection of the distribution.
        """
        return self.mu - self.sig*np.sqrt(self.W_z())

    def _calc_sig(self) -> float:
        """
        rearrangement of Equation 8 in Carreau and Bengio (2008)        

        Again, the valence is changed
        """
        return (self.mu - self.a) / np.sqrt(self.W_z())

    @argsetter('x')
    def pdf(self, x:XTYPE=None) -> np.ndarray:
        """
        Probability density function

        See Docs for derivation
        
        Parameters
        ----------
        x:      float, m x n iterable of floats of random variable samples

        Return
        -------
        float or iterable of floats returning the likelihood of occurence of each sample        
        """
        pdf = np.where(
            x < self.a,
            self.tail.pdf(-x),
            self.body.pdf(x)            
        )
        return pdf / self.gamma
    
    @argsetter('x')
    def cdf(self, x:XTYPE=None) -> np.ndarray:
        """
        Cumulative Distribtion Function, P(X<x)

        See Docs for derivation

        Parameters
        ----------
        x:      float, m x n iterable of floats of random variable samples

        Return
        -------
        float or iterable of floats returning cumulative probability of each sample
        """
        cdf = np.where(
            x < self.a,
            (1 - self.tail.cdf(-x)),
            1 + self.body.cdf(x) - self.body.cdf(self.a),
        )
        return cdf / self.gamma
        
    @property
    def qjunc(self):
        """
        Quantile of the juncture point, a, between the tail and body
        
        See Docs for derivation
        """
        return 1 / self.gamma
    
    def ppf(self, q:XTYPE) -> np.ndarray:
        """
        Quantile function, analogous to `ppf` method in `scipy`.

        See Docs for derivation

        Parameters
        ----------
        q:      float, iterable of floats between [0, 1]

        Return
        -------
        float or iterable of floats returning x values corresponding to the given quantiles
        """
        ppf = np.where(
            q < self.qjunc,
            -self.tail.ppf(1 - q*self.gamma),
            self.body.ppf(self.gamma*q + self.gamma*self.qjunc - self.gamma)
        )
        return ppf

    def mean(self) -> float:
        """
        First moment

        See Docs for derivation
        """
        bmu = scist.truncnorm(
            self.a,
            np.inf, 
            *self.body.args
        ).mean()
        tmu = -self.tail.mean()
        lmu = (bmu + tmu) / self.gamma

        return lmu

    def var(self) -> float:
        """
        Second moment

        See Docs for derivation
        """
        bvar = scist.truncnorm(
            self.a,
            np.inf, 
            *self.body.args
        ).var()
        tvar = self.tail.var()
        lvar = (bvar + tvar) / self.gamma
        return lvar

class CarBenHybrid: 
    """
    Light wrapper that simply reads the args/kwargs and
    selects the tail to be used, either CarBenLeft or CarBenRight

    If `xi>=0`:
        if `rtail` argument == True, use CarBenRight
        if `rtail` argument == False, use CarBenLeft
    elif `xi<0`:
        use CarBenLeft

    i.e. You can simply pass a negative tail index to return a left-skewed CarBen
    """  
    def __new__(cls, *args, **kwargs):
        """
        Takes arguments as per CarBenLeft/CarBenRight/CarBenBase

        Returns 
        --------
        CarBenLeft or CarBenRight object
        """
        args = list(args)
        xi_is_arg =  len(args) >= 0
        xi = args[0] if xi_is_arg else kwargs['xi']
        
        rtail_is_arg = len(args) >= 5
        if rtail_is_arg:
            rtail = args.pop(4)
        else:
            rtail = kwargs.pop('rtail') if 'rtail' in kwargs else True

        if isinstance(xi, float):
            if xi < 0 and not rtail:
                txt = 'If you provide a negative xi parameter,'
                txt += ' do not provide `rtail`.'
                txt += ' A left-tailed CarBen is computed automatically.'
                raise ValueError(txt)             
            elif xi < 0:
                rtail = False
                if xi_is_arg:
                    args[0] = -xi
                else:
                    kwargs['xi'] = xi

        if rtail:
            obj = CarBenRight(*args, **kwargs)
        else:
            obj = CarBenLeft(*args, **kwargs)
            
        return obj

class PhatFit(GenericLikelihoodModel):
    """
    Light wrapper for statsmodels GenericLogLikelihood. Used to mirror scipy or 
    statsmodels fit() method.

    Parameters
    -----------
    endog:      iterable of dependet variable
    exog:       iterable of independet variable
    xi_l:       inverse of tail index of left-tailed generalized Pareto tail. Default: None
    xi_r:       inverse of tail index of right-tailed generalized Pareto tail. Default: None
    tail_est:   func; process used to estimate both left and right tail indices. Default: None

    Attributes
    -----------
    left:               left-tailed CarBenHybrid object
    right:              right-tailed CarBenHybrid object
    args:               list of values of 8 parameters of Phat distribution
    params:             name-value pairs of 8 parameters of Phat distribution
    learnable_params:   name-value pairs of 4 given parameters of Phat distribution

    Methods
    --------
        > handle helper and common distribution functions that are agnostic to tail-direction
    """
    def __init__(self, 
        endog:Iterable, 
        exog:Iterable,
        xi_left:float=None, 
        xi_right:float=None, 
        tail_est:Callable=None,
        **kwds):
        
        if xi_right is None and xi_left is None and tail_est is not None:
            self.xi_left, self.xi_right = tail_est(endog)
        else:
            self.xi_left, self.xi_right = xi_left, xi_right
            
        super().__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        """
        Adjust Phat object parameters based on arguments provided at instantiation
        """
        if len(params) == 4:
            phatdist = Phat(*params)
        else:
            phatdist = Phat(params[0], params[1], self.xi_left, self.xi_right)
        return phatdist.nll(self.endog)

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        """
        Determines start parameters of fit method based on arguments provided at instantiation

        start_params for Gaussian are set via scipy `norm` distribution.
        """
        if start_params == None:
            if self.xi_left is None and self.xi_right is None:
                self.exog_names.append('xi_l')
                self.exog_names.append('xi_r')
                start_params = np.zeros(4)
                start_params[:2] = scist.norm.fit(self.endog)
                start_params[-2:] = [.2, .2]
            else:
                start_params = np.zeros(2)
                start_params = scist.norm.fit(self.endog)
            
        return super().fit(
            start_params=start_params,
            maxiter=maxiter, maxfun=maxfun,
            **kwds
        )

class Phat:
    """
    Twin-tailed distribution that combines a right-tailed CarBen with a left-tailed CarBen

    Parameters:
        mu:    
            mean of the Gaussian body
        sig:
            standard deviation of the Gaussian body
        xi_l:   
            inverse of tail index of left-tailed generalized Pareto tail
        xi_r:   
            inverse of tail index of right-tailed generalized Pareto tail
        p:
            relative weights of the two CarBen hybrids. Default: [.5, .5]

    Attributes:
        left:               
            left-tailed CarBenHybrid object
        right:              
            right-tailed CarBenHybrid object

    **Details**

    Per Carreau and Bengio (2008).
    
    The properites of the body in both distributions are equivalent; thus, the mixture requires just
    four parameters: two parameters for the Gaussian body and the tail index for each of the two tails. 
    In all, the distribution has 8 parameters:

    + Left tail: xi, a, b
    + Right tail: xi, a, b
    + Center: mu, sig

    As with any mixture model, this mix is the weighted average result of values from the two
    components. Default weights are 0.5 / 0.5.

    Weights could otherwise be estimated via machine learning, threshold analysis (for the tails),
    and gaussian statistics (for the body).
    """
    PARAM_NAMES = ['mu', 'sig', 'xi_l',  'xi_r', 'a_l', 'a_r', 'b_l', 'b_r']

    def __init__(
        self, mu:float, sig:float, xi_l:float, xi_r:float, p:Iterable=None
        ) -> None:
        """
        Left and right tails are instantiated immediately
        """
        self.mu = mu
        self.sig = sig
        self.xi_l = xi_l if xi_l >= 0 else -xi_l
        self.xi_r = xi_r

        self.left = CarBenHybrid(xi_l, mu, sig, rtail=False)
        self.right = CarBenHybrid(xi_r, mu, sig)

        self.p = p if p is not None else np.array([.5,.5])
    
    @property
    def args(self) -> tuple:
        """
        Tuple of the distribution parameters
        """
        return self.mu, self.sig, self.xi_l, self.xi_r, self.left.a, \
            self.right.a, self.left.b, self.right.b

    @property
    def params(self) -> dict:
        """
        Dictionary of the distribution parameters
        """      
        return {name: arg for name, arg in zip(self.PARAM_NAMES, self.args)}

    @property
    def learnable_params(self) -> List[float]:
        """
        List of distribution parameters that can be tuned by machine learning approach
        """
        return [self.params[name] for name in self.PARAM_NAMES[:4]]

    @argsetter('x')
    @dotweight
    def pdf(self, x:XTYPE=None) -> np.ndarray:
        return stacker(x)((self.left.pdf(x), self.right.pdf(x)))
    
    @argsetter('x')
    @dotweight    
    def cdf(self, x:XTYPE=None) -> np.ndarray:
        return stacker(x)((self.left.cdf(x), self.right.cdf(x)))
    
    @argsetter('x')
    def sf(self, x:XTYPE=None) -> np.ndarray:
        return 1 - self.cdf(x)
    
    @dotweight
    def ppf(self, q:XTYPE=None) -> np.ndarray:
        """
        Parameters:
            q: float, iterable of floats 
                Between [0, 1]
        """        
        return stacker(q)((self.left.ppf(q), self.right.ppf(q)))

    def loglike(self, x=None):
        "Loglikelihood"
        return np.log(self.pdf(x))

    @argsetter('x')
    def nll(self, x:XTYPE=None) -> np.ndarray:
        """Negative loglikelihood used in machine learning cost functions"""
        return -np.log(self.pdf(x))

    def rvs(self, 
        size:Union[int, Iterable[Sequence[int]]]=None, 
        seed:int=None) -> XTYPE:
        """
        Generates random draws

        **Process**

        1. Draws randomly from [0,1] uniform distribution
        2. Calls quantile function for each draw

        Mimics `scipy process <https://github.com/scipy/scipy/blob/28e4811d48c99b7c68e41992eda5cff859c1fa2b/scipy/stats/_distn_infrastructure.py#L1032>`_.
        """
        random_state = check_random_state(seed)
        U = random_state.uniform(size=size)
        if size is None:
            U = np.array(U, ndmin=1)
            Y = self.ppf(U)[0]
        else:
            Y = self.ppf(U)
        
        return Y

    @dotweight
    def mean(self) -> float:
        return np.array([self.left.mean(), self.right.mean()])

    @dotweight
    def var(self) -> float:
        return np.array([self.left.var(), self.right.var()])

    def std(self) -> float:
        return np.sqrt(self.var())

    def std_rvs(self, *args, **kwargs) -> XTYPE:
        """
        Generates random variables standardized by dividing by the standard deviation

        Resulting samples should have a standard deviation of 1
        """
        return self.rvs(*args, **kwargs) / self.std()

    @staticmethod
    def fit(values, xi_l:float=None, xi_r:float=None) -> PhatFit:
        """
        Generates standard log-likelihood fit to Phat distribution

        Returns:
            PhatFit
        """
        exog = sm.add_constant(np.zeros_like(values), prepend=True)
        phatfit = PhatFit(values, exog, xi_l, xi_r)

        return phatfit.fit()  

class PhatStack(Phat):
    """
    Weighted model of multiple Phat distributions. 
    
    This is NOT a mixture model. Used only to generate a two dimensional set
    of random variables that are consistent in a single component
    distribution across the second dimension.
    
    Parameters
    -----------
    phats:  iterable or args Phat distributions
    p:      iterable of floats; represents weighting of each Phat distribution
            in the final model. Must sum to 1.

    Attributes
    -----------
    n_comps:      int; number of component phat distributions
    
    Methods
    --------
    rvs
    
    """    
    def __init__(
        self, 
        *phats:Phat,
        p:Iterable=None
        ) -> None:
        """
        """
        if len(phats) == 1:
            phats = phats[0]
        
        if len(phats) != len(p):
            txt = 'Length of probabilities does not match number' 
            txt += f' of distributions: {len(phats)} v {len(p)}'
            raise AssertionError(txt)
        
        self.n_comps = len(phats)
        self.phats = phats
        self.p = arrayarize(p)

    def rvs(self, 
        size:Iterable[int]=None, 
        seed:int=None,
        return_splits:bool=False,
        ) -> XTYPE:
        """
        Generates array of random draws.
        
        ***IMPORTANT***
        Accepts only 2D m x n array. Draws are taken proportionately from
        the component Phat distributions ONLY along the first axis. Along the 
        second axis, draw are taken ONLY from the specific Phat distribution 
        isolated in the first axis.
        
        This ensures that only one distribution regime impacts the time series
        over time.
        
        Where the required iterations, m, does not divide evenly into the
        proportions, the remainders are added to certain distributions

        Params
        -------
        size:           iterable; m x n. Only two dimensions allowed.
        seed:           int; initialize with specific random state for replication
        return_splits:  bool; optional return of the number of m iterations
                        returned for each component distribution
        """
        if len(size) != 2:
            raise AssertionError('Size must be 2D m x n array')

        iters = np.floor(size[0] * self.p).astype(int)
        rmdr = size[0] - iters.sum()
        iters[:rmdr] += np.ones(rmdr, dtype=int)
        
        Y = np.zeros(size)
        st = 0
        for i, end in enumerate(iters):
            Y[st: st+end] = self.phats[i].rvs((end, size[1]), seed=seed)
            st += end
        
        if return_splits:
            return Y, iters
        else:
            return Y
            