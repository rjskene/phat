import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import tensorflow as tf

logger = tf.get_logger()

def hill_est(data, return_details=False):
    """
    Data is sorted in reverse order (i.e. kth largest first)
    
    Adapted from: https://github.com/ivanvoitalov/tail-estimation
    """
    logs = tf.math.log(data)
    k = tf.range(1., data.shape[0], dtype=tf.float64)
    M1 = (tf.constant(1., dtype=tf.float64)/k)*tf.math.cumsum(logs[:-1]) - logs[1:]
    return M1, logs, k

def second_moment(data, logs1, k=None):
    if k is None:
        k = tf.range(1., tf.shape(data)[0])
    
    logs2 = tf.math.log(data)**tf.constant(2., dtype=tf.float64)
    term1 = (1/tf.cast(k, dtype=np.float64))*tf.math.cumsum(logs2[:-1])
    term2 = (tf.constant(2., dtype=tf.float64)*logs1[1:]/tf.cast(k, dtype=np.float64))*tf.math.cumsum(logs1[:-1])
    M2 = term1 - term2 + logs2[1:]
    
    return M2
    
def moments_estimator(data):
    """
    Function to calculate second moments array
    given an ordered data sequence. 
    Decreasing ordering is required.
    Args:
        ordered_data: numpy array of ordered data for which
                      the 1st (Hill estimator) and 2nd moments 
                      are calculated.
    Returns:
        M2: numpy array of 2nd moments corresponding to all 
            possible order statistics of the dataset.
    """

    M1, logs1, k = hill_est(data, return_details=True)
    M2 = second_moment(data, logs1, k)
    return M1, M2

def amse(M1, M2):
    """
    Asymptotic Mean Squared Error derived from first and second moments
    
    M1 is adjusted to account for different multiplicative constants in the two moments
    """
    return (M2 - 2*M1**2)**2

def bootstrap(data, n, r):
    errs = tf.zeros((r, n-1), dtype=tf.float64)
    valid = tf.zeros((r, n-1), dtype=tf.float64)
    for i in range(r):
        sample = tf.numpy_function(np.random.choice, [data, n, tf.constant(True)], tf.float64)
        # sample = np.random.choice(data, n, replace=True)
        sample = tf.sort(sample, direction='DESCENDING')
        M1, M2 = moments_estimator(sample)
        
        err = M2 - tf.constant(2, dtype=tf.float64)*M1**tf.constant(2, dtype=tf.float64)
        val = tf.cast(~tf.math.is_nan(errs[i]), dtype=tf.float64)

        idxs_to_update = tf.stack((tf.repeat(i, n-1), tf.range(n-1)), axis=1)
        tf.tensor_scatter_nd_update(errs, idxs_to_update, err)
        tf.tensor_scatter_nd_update(valid, idxs_to_update, val)

    return tf.math.reduce_sum(errs, axis=0)**tf.constant(2, dtype=tf.float64) / tf.math.reduce_sum(valid, axis=0)

def k_finder(data, idx, n, r, t):
    avg_amses = bootstrap(data[idx:], n, r)
    return tf.cast(tf.math.argmin(avg_amses), dtype=tf.int32) + tf.constant(1, dtype=tf.int32) + idx

def rho_dani(n1, k1):
    return (tf.math.log(k1)/(2.*np.log(n1) - np.log(k1)))\
          **(2.*(np.log(n1) - np.log(k1))/(np.log(n1)))

def rho_qi(n1, k1):
    return (tf.constant(1, dtype=tf.float64) - (tf.constant(2, dtype=tf.float64)*(tf.math.log(k1) - tf.math.log(n1))/(tf.math.log(k1))))**(tf.math.log(k1)/tf.math.log(n1) - tf.constant(1, dtype=tf.float64))

def hill_dbl_bs(data, t=.5, r=500, rho_type='qi'):
    """
    Different research provides different \rho values. we use
    rho_dani constant is given in Corollary 7 of Danielsson's paper
    https://www.sciencedirect.com/science/article/pii/S0047259X00919031
    rho_qi 
    https://link.springer.com/article/10.1007%2Fs10687-007-0049-8
    
    # eps_stop = 1
    # eps = 0.5*(1+np.log(int(t*n))/np.log(n)) # ???????
    # max_index1 = (np.abs(np.linspace(1/n1, 1.0, n1) - eps_stop)).argmin()
    """
    t = tf.cast(t, dtype=tf.float64)
    n = tf.cast(tf.size(data), dtype=tf.float64)
    xi, _, __ = hill_est(data)
    if logger.level <= 10:
        tf.print('Init Hill Estimates: ', xi[0])
    min1 = min2 = tf.constant(1, dtype=tf.int32)
    n1 = tf.cast(tf.multiply(tf.math.sqrt(t), n), dtype=tf.int32)
    n2 = tf.cast(tf.multiply(t, n), dtype=tf.int32)
    k1 = k2 = tf.constant(0, dtype=np.int32)
    
    for i in tf.range(tf.size(data)):
        k1 = k_finder(data, min1, n1, r, t)
        k2 = k_finder(data, min2, n2, r, t)
     
        if k2 > k1:
            min1 += tf.cast(tf.multiply(tf.constant(0.005, dtype=tf.float64), tf.cast(n1, dtype=tf.float64)), dtype=tf.int32)
            min2 += tf.cast(tf.multiply(tf.constant(0.005, dtype=tf.float64), tf.cast(n1, dtype=tf.float64)), dtype=tf.int32)
        else:
            break

    if rho_type == 'dani':
        rho = rho_dani(tf.cast(n1, dtype=tf.float64), tf.cast(k1, dtype=tf.float64)) 
    elif rho_type =='qi':
        rho = rho_qi(tf.cast(n1, dtype=tf.float64), tf.cast(k1, dtype=tf.float64)) 
    else:
        raise ValueError("`rho_type` must be either 'dani' or 'qi'")
    
    tf.print('Ns', n, n1, n2)
    tf.print('k1', k1, 'k2', k2)
    k_star = tf.cast(tf.math.round(rho*tf.cast(k1, dtype=tf.float64)**tf.constant(2, dtype=tf.float64)/tf.cast(k2, dtype=tf.float64)), dtype=tf.int32)

    if k_star >= tf.cast(n, dtype=tf.int32):
        tf.print('k1', k1, 'k2', k2)
        tf.print("WARNING: estimated threshold k is larger than the size of data")
        k_star = tf.cast(n, dtype=tf.int32)-tf.constant(1, dtype=tf.int32)
    elif k_star == tf.constant(0, dtype=tf.int32):
        k_star = tf.constant(2, dtype=tf.int32)
    else:
        k_star = k_star

    return xi[k_star-1], k_star

def conc_circles(alphas, ax=None, u50=False):
    if ax is None:
        fig, ax = plt.subplots()
    
    n_u10 = (np.abs(alphas[:,2]) < .1).sum()
    u10_per = n_u10 / alphas.shape[0]
    n_u25 = (np.abs(alphas[:,2]) < .25).sum()
    u25_per = n_u25 / alphas.shape[0]
    n_u50 = (np.abs(alphas[:,2]) < .5).sum()
    u50_per = n_u50 / alphas.shape[0]

    ax.add_patch(mpatches.Circle(
        (.5,0), 1, fc='grey', ec='grey', 
        ls='--', lw=1, alpha=.5, zorder=3
    ))
    ax.add_patch(mpatches.Circle((.5,0), u25_per, fc='C1', zorder=5))
    ax.add_patch(mpatches.Circle((.5,0), u10_per, fc='C0', zorder=6))

    ax.text(.5, .9, f'Total: {alphas.shape[0]:.0f}', ha='center')
    ax.text(.5, .75*u25_per, f'Error <25%: {n_u25:.0f}, {u25_per:.0%}', ha='center', zorder=5)
    ax.text(.5, .75*u10_per, f'Error <10%: {n_u10:.0f}, {u10_per:.0%}', ha='center', zorder=6)

    if u50:
        ax.add_patch(mpatches.Circle((.5,0), u50_per, fc='C2', zorder=4))
        ax.text(.5, .75*u50_per, f'Error <50%: {n_u50:.0f}, {u50_per:.0%}', ha='center', zorder=4)

    ax.axis('off')
    ax.set_title('Accuracy of Hill Double BootStrap')
