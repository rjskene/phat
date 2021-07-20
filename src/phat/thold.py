from functools import wraps
from typing import Iterable

import numpy as np
import scipy.stats as scist
import matplotlib.pyplot as plt

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

from phat.utils import argsetter

base = importr('base')
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('POT')
POT = importr('POT')

def fit_line_within(stacked, ival):
    ivalmask = np.logical_and(stacked[:,0]>=ival[0], stacked[:,0]<=ival[1])
    return (*scist.linregress(stacked[ivalmask])), ivalmask.sum()

def threshset(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        data = kwargs['data']
        spacer = 45 if 'spacer' not in kwargs else kwargs['spacer']
    
        if not hasattr(self, 'tholds') and 'tholds' not in kwargs:
            step = np.quantile(data, .995)/spacer
            tholds = np.arange(-.1, max(data), step=step)
            self.tholds = tholds
        elif 'tholds' not in kwargs:
            tholds = self.tholds
        else:
            self.tholds = kwargs['tholds']

        kwargs['tholds'] = tholds
    
        return func(self, *args, **kwargs)
    
    return wrapper


class Threshold:
    def __init__(self, data):
        self.data = data
        
    @argsetter(['data'])
    @threshset
    def MRL(self, data:Iterable=None,
            tholds:Iterable=None, alpha:float=.05,
            fig=None, ax=None, show_plot:bool=True,
            splits:Iterable=None, *args, **kwargs
        ):
        is_excess = np.array([data > thold for thold in tholds])
        excesses = np.array([data - thold for thold in tholds])
        excesses = np.where(
            is_excess,
            excesses,
            np.nan
        )

        self.mean_exc = np.nanmean(excesses, axis=1)
        stds = np.nanstd(excesses, axis=1)
        z_inverse = scist.norm.ppf(1-(alpha/2))
        CI = z_inverse*stds/(len(excesses)**0.5)
        
        if show_plot:
            if fig is None or ax is None:
                fig, ax = plt.subplots(1,1,figsize=(10,6))
            
            ax.plot(tholds, self.mean_exc)
            ax.fill_between(tholds, self.mean_exc - CI, self.mean_exc + CI, alpha = 0.4)
            ax.set_xlabel('u')
            ax.set_ylabel('Mean Excesses')
            ax.set_title('Mean Residual Life Plot')

        if splits is not None:
            self._MRL_regrs(splits, ax)

    def _MRL_regrs(self, splits:Iterable, ax):
        splits = np.array(splits)
        stacked = np.vstack([self.tholds, self.mean_exc]).T
        
        sgmts = np.vstack((splits[:-1], splits[1:])).T
        for i in range(sgmts.shape[0]):
            sgmt = sgmts[i]
            b, a, r, p, stderr, n = fit_line_within(stacked, sgmt)

            count = (self.data>sgmt[1]).sum()
            y = b*sgmt + a
            label = '[{:.4f},{:.4f}] N<{}; N>{}'.format(*sgmt, n, count) + r' $R^2: $' + f'{r**2:.0%}'
            label += f' p-value: {p:.02f}'
            ax.plot(sgmt, y, label=label)

        plt.legend(loc='best')

    @argsetter(['data'])
    @threshset
    def param_stable(self, data:Iterable=None,
            tholds:Iterable=None, alpha:float=.05,
            fig=None, axs=None,
            *args, **kwargs
        ):
        shape = []
        scale = []
        mod_scale = []
        CI_shape = []
        CI_mod_scale = []
        z = scist.norm.ppf(1-(alpha/2)) 
        for thold in tholds:
            fit, _, _ = self.fit(data=data, thold=thold.item(), est='mle')

            shape.append(fit[0][1])
            CI_shape.append(fit[1][1]*z)

            scale.append(fit[0][0])
            mod_scale.append(fit[0][0] - (fit[0][1]*thold))
            Var_mod_scale = (fit[3][0] - (thold*fit[3][2]) - thold*(fit[3][1] - (fit[3][3]*thold))) 
            CI_mod_scale.append((Var_mod_scale**0.5)*z)

        #Plotting shape parameter against u vales   
        axs[0].errorbar(tholds, shape, yerr = CI_shape, fmt = 'o' )
        axs[0].set_xlabel('u')
        axs[0].set_ylabel('Shape')
        axs[0].set_title('Shape Parameter Stability')

        #Plotting modified scale parameter against u values
        axs[1].errorbar(tholds, mod_scale, yerr = CI_mod_scale, fmt = 'o')
        axs[1].set_xlabel('u')
        axs[1].set_ylabel('Modified Scale')
        axs[1].set_title('Modified Scale Parameter Stability')

    @argsetter(['data'])
    def fit(self, data:Iterable=None, thold:float=0, est:str='mle'):
        rdata = np.sort(data)   
        data_over_thresh = rdata[rdata > thold]
        data_exc= data_over_thresh - thold

        rdata = FloatVector(rdata)
        fit = POT.fitgpd(rdata, thold, est=est)
        
        return fit, data_over_thresh, data_exc

    @argsetter(['data'])
    def qqplot(self, 
        data:Iterable=None, thold:float=0, est:str='mle', alpha:float=.05,
        fig=None, ax=None
        ):
        fit, over_thresh, _ = self.fit(data=data, thold=thold, est=est)
        scale, shape = fit[0][0], fit[0][1]

        p = []
        n = len(data)
        data = np.sort(data)  
        i_initial = np.searchsorted(data, thold)
        k = i_initial - 1

        p = (np.arange(i_initial, n) - .35) / n
        p0 = (k - 0.35)/(n)  
        quantiles = thold + ((scale/shape)*(((1-((p-p0)/(1-p0)))**-shape) - 1))

        n = len(over_thresh)
        y = np.arange(1,n+1)/n

        #Kolmogorov-Smirnov Test for getting the confidence interval
        K = (-0.5*np.log(alpha/2))**0.5
        M = (len(p)**2/(2*len(p)))**0.5
        CI_qq_high = []
        CI_qq_low = []
        for prob in y:
            F1 = prob - K/M
            F2 = prob + K/M
            CI_qq_low.append(thold + ((scale/shape)*(((1-((F1)/(1)))**-shape) - 1)))
            CI_qq_high.append(thold + ((scale/shape)*(((1-((F2)/(1)))**-shape) - 1)))
        
        a, b, r_value, p_value, std_err = scist.linregress(quantiles, over_thresh)
        ax.scatter(quantiles, over_thresh)
        
        x = np.linspace(0,1,101)*100
        ax.plot(x, a*x + b, c='black', label='Regression')
        
        ax.plot(over_thresh, CI_qq_low, linestyle='--', color='red', alpha = 0.5, lw = 0.8, label='Confidence Bands')
        ax.plot(over_thresh, CI_qq_high, linestyle='--', color='red', alpha = 0.5, lw = 0.8)
        ax.set_xlabel('Theoretical GPD Quantiles')
        ax.set_ylabel('Sample Quantiles')

        ax.legend()
        ax.set_title('Q-Q Plot')

    @argsetter(['data'])
    def ppplot(self, data:Iterable=None, thold:float=0, est:str='mle', alpha:float=.05,
        fig=None, ax=None
        ):
        fit, over_thresh, _ = self.fit(data=data, thold=thold, est=est)
        scale, shape = fit[0][0], fit[0][1]
        
        n = len(over_thresh)
        y = np.arange(1,n+1)/n
        cdf_pp = scist.genpareto.cdf(over_thresh, shape, loc=thold, scale=scale)
        
        #Getting Confidence Intervals using the Dvoretzky–Kiefer–Wolfowitz method
        data = np.sort(data)
        i_initial = np.searchsorted(data, thold)
        
        F1 = []
        F2 = []
        for i in range(i_initial, len(data)):
            e = (((np.log(2/alpha))/(2*len(over_thresh)))**0.5)
            F1.append(y[i-i_initial] - e)
            F2.append(y[i-i_initial] + e)

        ax.scatter(y, cdf_pp)
        a, b, r_value, p_value, std_err = scist.linregress(y, cdf_pp)
        ax.plot(y, a*y + b, c='black', label='Regression')
        
        ax.plot(y, F1, linestyle='--', color='red', alpha = 0.5, lw = 0.8, label = 'Confidence Bands')
        ax.plot(y, F2, linestyle='--', color='red', alpha = 0.5, lw = 0.8)
        
        ax.set_xlabel('Empirical Probability')
        ax.set_ylabel('Theoritical Probability')
        ax.legend()
        ax.set_title('P-P Plot')

    @argsetter(['data'])
    def return_value(self, data:Iterable=None, thold:float=0, alpha:float=.05,
        block_size:int=252, return_period:int=252*100, est:str='mle',
        fig=None, ax=None
        ):
        data = np.sort(data) 
        fit, over_thresh, _ = self.fit(data=data, thold=thold, est=est)
        scale, shape = fit[0][0], fit[0][1]  
        
        #Computing the return value for a given return period with the confidence interval estimated by the Delta Method
        m = return_period
        Eu = len(over_thresh)/len(data)
        x_m = thold + (scale/shape)*(((m*Eu)**shape) - 1)

        #Solving the delta method    
        d = Eu*(1-Eu)/len(data)
        e = fit[3][0]
        f = fit[3][1]
        g = fit[3][2]
        h = fit[3][3]
        a = (scale*(m**shape))*(Eu**(shape-1))
        b = (shape**-1)*(((m*Eu)**shape) - 1)
        c = (-scale*(shape**-2))*((m*Eu)**shape - 1) + (scale*(shape**-1))*((m*Eu)**shape)*np.log(m*Eu)
        CI = (scist.norm.ppf(1-(alpha/2))*((((a**2)*d) + (b*((c*g) + (e*b))) + (c*((b*f) + (c*h))))**0.5))

        ny = block_size
        N_year = return_period/block_size
        i_initial = np.searchsorted(data, thold)

        p = np.arange(i_initial,len(data))/(len(data))
        N = 1/(ny*(1 - p))
        year_array = np.arange(min(N), N_year+0.1, 0.1)
        
        #Algorithm to compute the return value and the confidence intervals for plotting
        z_N = []
        CI_z_N_high_year = []
        CI_z_N_low_year = [] 
        for year in year_array:
            z_N.append(thold + (scale/shape)*(((year*ny*Eu)**shape) - 1))
            a = (scale*((year*ny)**shape))*(Eu**(shape-1))
            b = (shape**-1)*((((year*ny)*Eu)**shape) - 1)
            c = (-scale*(shape**-2))*(((year*ny)*Eu)**shape - 1) + (scale*(shape**-1))*(((year*ny)*Eu)**shape)*np.log((year*ny)*Eu)
            CIyear = (scist.norm.ppf(1-(alpha/2))*((((a**2)*d) + (b*((c*g) + (e*b))) + (c*((b*f) + (c*h))))**0.5))
            CI_z_N_high_year.append(thold + (scale/shape)*(((year*ny*Eu)**shape) - 1) + CIyear)
            CI_z_N_low_year.append(thold + (scale/shape)*(((year*ny*Eu)**shape) - 1) - CIyear)
        
        #Plotting Return Value
        ax.plot(year_array, CI_z_N_high_year, linestyle='--', color='red', alpha = 0.8, lw = 0.9, label = 'Confidence Bands')
        ax.plot(year_array, CI_z_N_low_year, linestyle='--', color='red', alpha = 0.8, lw = 0.9)
        ax.plot(year_array, z_N, color = 'black', label = 'Theoretical Return Level')
        ax.scatter(N, over_thresh, label = 'Empirical Return Level')

        
        text = f'{N_year:.0f} Year Return Level: {x_m:.2f} \u00B1 {CI:.2f}'
        ax.text(.6,.05,text, transform=ax.transAxes)

        ax.set_xscale('log')
        ax.set_xlabel('Return Period')
        ax.set_title('Return Level Plot')
        ax.legend()

