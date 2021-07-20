import numpy as np
import scipy.stats as scist
import scipy.special as spec
from scipy._lib._util import check_random_state

from functools import wraps

from phat.utils import argsetter, sech_squared

def karamata_from_call(call, alf):
    t1 = (alf-1)**(1/alf)
    t2 = call.prem**(1/alf)
    t3 = max(call.K - call.S.price,0)**(1-(1/alf))

    return t1*t2*t3/call.S.price

class BLineSig:
    def __init__(self, k:float=1):
        self.k = k

    @argsetter('xk')
    def cdf(self, x, k:float=None):
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
    def __init__(self, alpha=2, L=1):
        self.alpha = alpha
        self.L = L
        
    def sf(self, x):
        return (self.L**self.alpha)*(np.log(x)**(-self.alpha))

class DBLFat(scist.rv_continuous):
    @property
    def dist(self):
        raise NotImplementedError

    @property
    def q_junc(self):
        return .5

    @argsetter(['x', 'shape'])
    def _pdf(self, x, shape):
        pdf = np.where(
            x>=0, 
            self.dist.pdf(x, shape), 
            self.dist.pdf(-x, shape)
        )
        return pdf
    
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
            q>=self.q_junc,
            self.dist.ppf(1/self.q_junc*(q - self.q_junc), shape),
            -self.dist.isf(q*1/self.q_junc, shape)
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

class CarbenBase:
    """
    One-tailed assymetric hybrid distribution of Normal and Generalized pareto
    
    Takes 3 free parameters:
        > shape of Pareto, aka the tail index (inverse of Taleb's tail index)
        > mean, var of the Normal distribution
    
    Remaining 2 pareto parameters are found via equalities
        > scale, as function of var and shape (utilizing a Lambert W)
        > loc, can then be found either via a) scale OR b) via the Lambert W. the latter is used
    
    The two distributions are mixed using a weighting factor, gamma.
        > gamma determined via the Erf of the Lambert W
    
    https://www.researchgate.net/publication/226293435_A_hybrid_Pareto_model_for_asymmetric_fat-tailed_data_The_univariate_case#pf6
    """

    def __init__(self, shape, mean, sig:float=None, loc:float=None):
        self.shape = shape
        self.mean = mean

        provided = (sig is None, loc is None)
        if all(provided) or not any(provided):
            raise ValueError('Please provide only one of `sig` or `loc`')
        elif sig is not None:
            self.sig = sig
            self._given = 'sig'
            self.scale = self._calc_scale()
            self.loc = self._calc_loc()
        elif loc is not None:
            self.loc = loc
            self._given = 'loc'
            self.scale = self._calc_scale()
            self.sig = self._calc_sig()
                
        self.body = scist.norm(self.mean, self.sig)
        
    @property
    def tail(self):
        if hasattr(self, '_tail'):
            return self._tail
        else:
            raise NotImplementedError

    def z(self):
        """
        in Carreau (2008), z defined as:

            (1 + shape)**2 / 2*pi
        """
        num = (1 + self.shape)**2
        denom = 2*np.pi
        return  num / denom        

    def W_z(self):
        """
        in Carreau (2008), W(z) can be calculated directly as:
            
            sig**2 * (1 + shape)**2 / scale**2

        scale is not a free parameter, however, so we must calculate W(z) via
        the Lambert function on z, which is defined only in free parameters.
        """

        val = spec.lambertw(self.z())
        assert (val.imag == 0).all(), val.imag

        return val.real

    @property
    def gamma(self):
        val = np.sqrt(self.W_z() / 2)
        return 1 + .5*(1 + spec.erf(val))

    def _calc_scale_w_sig(self):
        num = self.sig*(1 + self.shape)
        denom = np.sqrt(self.W_z())
        
        return num / denom

    def _calc_scale_w_loc(self):
        num = self.loc - self.mean*self.shape
        denom = np.sqrt(self.W_z())
        return num / denom

    def _calc_scale(self):
        return self._calc_scale_w_loc() if self._given == 'loc' else self._calc_scale_w_sig()

    @argsetter('x')
    def sf(self, x=None):
        return 1 - self.cdf(x)
    
    def rvs(self, size=None, seed=None):
        random_state = check_random_state(seed)

        U = random_state.uniform(size=size)
        if size is None:
            U = np.array(U, ndmin=1)
            Y = self.ppf(U)[0]
        else:
            Y = self.ppf(U)
        return Y

class CarbenRight(CarbenBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tail = scist.genpareto(self.shape, loc=self.loc, scale=self.scale)
    
    def _calc_loc(self):
        return self.mean + self.sig*np.sqrt(self.W_z())

    def _calc_sig(self):
        return (self.loc - self.mean) / np.sqrt(self.W_z())

    @argsetter('x')
    def pdf(self, x=None):
        """
        Right tail only.
        where x>a,
            f(x) = f_t(x) / gamma
        """        
        pdf = np.where(
            x > self.loc,
            self.tail.pdf(x),
            self.body.pdf(x)
        )
        return pdf / self.gamma
    
    @argsetter('x')
    def cdf(self, x=None):
        """
        where x>a,
            P(X<x) = (P_b(X=a) + P_t(X<x)) / gamma
        where x<=a,
            P(X<x) = P_b(X<x) / gamma

        *** could just replace with  self.body.cdf(self.loc) + self.tail.cdf(x) since for x<self.loc slef.tail.cdf(x) = 0
        """
        cdf = np.where(
            x > self.loc,
            self.body.cdf(self.loc) + self.tail.cdf(x),
            self.body.cdf(x)
        )
        return cdf / self.gamma    

    @property
    def q_junc(self):
        """
        Juncture point b/w body and tail
        
        Carben defines junction as:
            P(X>a) = 1 / gamma

        For a right skew, this is the proability in tail.
        
        Quantiles, however, are measured left to right, so the junction quantile must
        be determined from the probability in the body.    
            P(X<=a) = 1 - P(X>a) = 1 - 1/gamma
        """
        return 1 - 1 / self.gamma

    @argsetter('q')
    def ppf(self, q):
        """
        q, quantile = P(X<x)
        q_junc = P_b(X=a)

        where x>a, P(X>x) > P(X>a), in the tail which is body + tail
            
            Eq (1) from Carben: F(u+y) = F(u)+ (1−F(u))Fu(y)

            F(x) = F(a) + F_a(x) / gamma
                  body  +   weighted tail
 
            P(X<x) = P(X<=a) + P_t(X<x) / gamma
            P(X<=a) = P_b(X<=a)
            P(X<x) - P_b(X<=a) = P_t(X<x) / gamma
            P_t(X<x) = (P(X<x) - P_b(X<=a)) * gamma
            x = Q_t(P_t(X<x)) = Q_t(gamma*(P(X<x) - P_b(X<=a)))
            x = Q_t(gamma*(q - q_junc))
        where x<=a, P(X>x) <= P(X>a), in the body and ONLY in the body
            P(X<x) = P_b(X<x) / gamma
            P_b(X<x) = P(X<x)*gamma
            x = Q_b(P_b(X<x)) = Q_b(P(X<x)*gamma)
            x = Q_b(gamma*q)

        F_l(x) = 1 - F_r(-x)
        F_l(x) = 1 - F(a) + F_a(x) / gamma
        """
        ppf = np.where(
            q > self.q_junc,
            self.tail.ppf(self.gamma * (q - self.q_junc)),
            self.body.ppf(self.gamma*q)
        )
        return ppf    

    def var(self):
        bvar = scist.truncnorm(
            -np.inf, 
            self.loc, 
            *self.body.args
        ).var()
        tvar = self.tail.var()
        rvar = (bvar + tvar) / self.gamma
        return rvar

class CarbenLeft(CarbenBase):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tail = scist.genpareto(self.shape, loc=-self.loc, scale=self.scale)

    def _calc_loc(self):
        return self.mean - self.sig*np.sqrt(self.W_z())

    def _calc_sig(self):
        return (self.mean - self.loc) / np.sqrt(self.W_z())

    @argsetter('x')
    def pdf(self, x=None):
        """
        """
        pdf = np.where(
            x < self.loc,
            self.tail.pdf(-x),
            self.body.pdf(x)            
        )
        return pdf / self.gamma
    
    @argsetter('x')
    def cdf(self, x=None):
        cdf = np.where(
            x < self.loc,
            (1 - self.tail.cdf(-x)),
            self.gamma + self.body.cdf(x) - 1,
        )
        return cdf / self.gamma
        
    @argsetter('x')
    def sf(self, x=None):
        return (1 - self.cdf(x=x))

    @property
    def q_junc(self):
        """
        Juncture point b/w body and tail
        
        For Left tail, we DO use the definition from Carben, as we want probability
        of area in the tail:
            P_l(X<=-a) = P_lt(X<=-a) / gamma = (1 - P_rt(X<=a)) / gamma = 1 / gamma
        """
        return 1 / self.gamma
    
    @argsetter('q')
    def ppf(self, q):
        """
        WE GET q_l, NOT q_r
        q, quantile = P_l(X<x)
        q_junc = P_b(X=-a)
        For Left Tail:
            F_l(x) = 1 - F(-x)
            P_l(X<x) = 1 - P_r(X<-x)
            P_lt(X<x) = 1 - P_rt(X<-x)
            P_lb(X<x) = P_rb(X<x)
        so, where x<-a, P_l(X<x) < P_l(X<-a), in the tail and ONLY in the tail
            P_lt(X<x_l) = (1 - P_rt(X<x_r))
            P_l(X<x_l) =  (1 - P_rt(X<x_r)) / gamma
            q_l*gamma = 1 - P_rt(X<x_r)
            P_rt(X<x_r) = 1 - q_l*gamma
            x_r = Q_rt(1 - q_l*gamma)
            xl = -x_r = -Q_rt(1 - q_l*gamma)
        and, where x_l>=-a, P(X<=x_l) >= P(X>=-a), in the body, which is tail + body
                x_l = Q_b(-gamma*(1 - q_l - (1 / gamma))
                P_b(X<x_l) = -gamma*(1 - q_l - (1/gamma))
                P_b(X<x_l) = -gamma + gamma*q_l + 1
                q_l*gamma = P_b(X<x_l) + gamma - 1
                q_l = P_b(X<x_l)/gamma - 1/gamma + 1
                q_l = P_b(X<x_l)/gamma - P(X<-a)/gamma + P(X<-a)
                """
        ppf = np.where(
            q < self.q_junc,
            -self.tail.ppf(1 - q*self.gamma),
            self.body.ppf(-self.gamma*(1-q-self.q_junc))
        )
        return ppf

    def var(self):
        bvar = scist.truncnorm(
            self.loc,
            np.inf, 
            *self.body.args
        ).var()
        tvar = self.tail.var()
        lvar = (bvar + tvar) / self.gamma
        return lvar

class CarbenHybrid:    
    def __new__(cls, *args, **kwargs):
        args = list(args)
        shape_is_arg =  len(args) >= 2
        shape = args[1] if shape_is_arg else kwargs['shape']
        
        rtail_is_arg = len(args) >= 6
        if rtail_is_arg:
            rtail = args.pop(5)
        else:
            rtail = kwargs.pop('rtail') if 'rtail' in kwargs else True

        if isinstance(shape, float):
            if shape < 0 and not rtail:
                txt = 'If you provide a negative shape parameter,'
                txt += ' do not provide `rtail`'
                txt += ' as a left-tailed Carben is computed automatically.'
                raise ValueError(txt)             
            elif shape < 0:
                rtail = False
                if shape_is_arg:
                    args[1] = -shape
                else:
                    kwargs['shape'] = shape

        if rtail:
            obj = CarbenRight(*args, **kwargs)
        else:
            obj = CarbenLeft(*args, **kwargs)
            
        return obj

def dotweight(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        stack = func(self, *args, **kwargs)
        if stack.ndim == 3:
            p = np.tile(self.p, (stack.shape[0],stack.shape[1],1))
            return np.multiply(p, stack).sum(axis=2)
        else:
            return (self.p @ stack)

    return wrapper

def stacker(arr):
    """
    arr isn numpy array
    """
    if arr.ndim == 1:
        return np.vstack
    elif arr.ndim == 2:
        return np.dstack
    else:
        raise ValueError(f'`arr` has {arr.ndim}')

class Phat:
    """    
    Two-tailed symmetric OR assymmetric hybrid distribution that combines:
        1) right-tailed Carben 
        with 
        2) left-tailed Carben

    This two-tailed approach was suggested by Carben, but does not appear to 
    have been implemented or studied anywhere.
    https://www.researchgate.net/publication/226293435_A_hybrid_Pareto_model_for_asymmetric_fat-tailed_data_The_univariate_case#pf6
    
    The properites of the body in both distributions are equivalent; thus, the mixture requires just
    one additional parameter: the shape parameter of the left-tailed Carben. In all, the distribution
    has 8 parameters:
        > Left tail: shape, loc, scale
        > Right tail: shape, loc, scale
        > Center: mean, sig

    As with any mixture model, this mix is the weighted average result of values from the two
    components. Default weights are 0.5 / 0.5.

    Weights can otherwise be estimated via machine learning, threshold analysis (for the tails),
    and normal statistics (for the body).

    https://www.cs.toronto.edu/~rgrosse/csc321/mixture_models.pdf
    """
    PARAM_NAMES = ['mean', 'sig', 'shape_l',  'shape_r', 'loc_l', 'loc_r', 'scale_l', 'scale_r']
    def __init__(self, mean:float, sig:float, shape_l:float, shape_r:float, p=None):
        """
        Reverse the mean in the Left tail so that both tails
        are centered around the same mean
        """
        self.mean = mean
        self.sig = sig
        self.shape_l = shape_l if isinstance(shape_l, float) and shape_l >= 0 else -shape_l
        self.shape_r = shape_r

        self.left = CarbenHybrid(shape_l, mean, sig, rtail=False)
        self.right = CarbenHybrid(shape_r, mean, sig)

        self.p = p if p is not None else np.array([.5,.5])
    
    @property
    def args(self):
        return self.mean, self.sig, self.shape_l, self.shape_r, self.left.loc, \
            self.right.loc, self.left.scale, self.right.scale

    @property
    def params(self):        
        return {name: arg for name, arg in zip(self.PARAM_NAMES, self.args)}

    @property
    def learnable_params(self):
        return [self.params[name] for name in self.PARAM_NAMES[:4]]

    @argsetter('x')
    @dotweight
    def pdf(self, x=None):
        return stacker(x)((self.left.pdf(x), self.right.pdf(x)))
    
    @argsetter('x')
    @dotweight    
    def cdf(self, x=None):
        return stacker(x)((self.left.cdf(x), self.right.cdf(x)))
    
    @argsetter('x')
    def sf(self, x=None):
        return 1 - self.cdf(x)
    
    @argsetter('q')
    @dotweight
    def ppf(self, q):
        return stacker(q)((self.left.ppf(q), self.right.ppf(q)))

    @argsetter('x')
    def nll(self, x=None):
        return -np.log(self.pdf(x))

    def rvs(self, size=None, seed=None):
        random_state = check_random_state(seed)
        U = random_state.uniform(size=size)
        if size is None:
            U = np.array(U, ndmin=1)
            Y = self.ppf(U)[0]
        else:
            Y = self.ppf(U)
        
        return Y

    @dotweight
    def var(self):
        return np.array([self.left.var(), self.right.var()])

    def std(self):
        return np.sqrt(self.var())

    def std_rvs(self, *args, **kwargs):
        return self.rvs(*args, **kwargs) / self.std()

