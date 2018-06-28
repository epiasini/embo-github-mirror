from __future__ import division
import numpy as np

from numba import jit
from scipy.stats import entropy

from .utils import p_dist, p_joint, p_cond, mi_x1x2_c


def empirical_bottleneck(x,y,numuniquex=0,numuniquey=0,maxbeta=5,iterations=100):
    """ Compute an IB curve for two empirical sequences x and y"""
    
    # Marginal, joint and conditional distributions required to calculate the IB
    px = p_dist(x,numuniquex)
    py = p_dist(y,numuniquey)
    pxy_j = p_joint(x,y)
    pxy_c = p_cond(px,py,pxy_j)
    pyx_j = p_joint(y,x)
    pyx_c = p_cond(py,px,pyx_j)
    # Mutual information
    mi = mi_x1x2_c(px,py,pxy_c)
    #Calculate the information bottleneck for different values of beta
    i_p,i_f,beta = IB(px,py,pyx_c,maxbeta,iterations)
    # Return array of ipasts and ifutures for array of different values of beta - mi should correspond to the saturation point
    return i_p,i_f,beta,mi

def IB(px,py,pyx_c,maxbeta=5,iterations=100):
    """
    Compute an Information Bottleneck curve

    px: marginal probability distribution for the past
    py: marginal distribution for the future
    maxbeta: the maximum value of beta to use to compute the curve
    iterations: number of iterations to use to for the curve to converge for each value of beta
    
    return vectors of ipast and ifuture (ips and ifs respectively) for different values of beta (bs)
    """
    
    bs = np.linspace(0.01,maxbeta,30) #value of beta
    pm = np.repeat(1./(len(px)),len(px))
    pym_c = np.random.rand(len(py),len(pm))+1e-4 # Starting point for the algorithm
    pym_c = pym_c/pym_c.sum(axis=0)
    ips = np.zeros(len(bs))
    ifs = np.zeros(len(bs))
    for bi in range(len(bs)):
        for i in range(iterations):
            pmx_c = p_mx_c(pm,px,py,pyx_c,pym_c,bs[bi])
            pm = p_m(pmx_c,px)
            pym_c = p_ym_c(pm,px,py,pyx_c,pmx_c)
        ips[bi] = mi_x1x2_c(pm,px,pmx_c)
        ifs[bi] = mi_x1x2_c(py,pm,pym_c)
    return ips,ifs,bs

#@jit
def p_mx_c(pm,px,py,pyx_c,pym_c,beta):
    """Update conditional distribution of bottleneck random variable given x.

    pm: marginal distribution p(M) - vector
    px: marginal distribution p(X) - vector
    py: marginal distribution p(Y) - vector
    pyx_c: conditional distribution p(Y|X) - matrix
    pym_c: conditional distribution p(Y|M) - matrix
    """
    
    pmx_c = np.zeros((len(pm),len(px))) # P(M|X) matrix to be returned
    for mi in range(len(pm)):
        for xi in range(len(px)):
            pmx_c[mi,xi] = pm[mi] * np.exp(-beta * entropy(pyx_c[:,xi], pym_c[:,mi], base=2))
    return pmx_c/pmx_c.sum(axis=0) #Normalize 


@jit
def p_mx_j(pm,px,pmx_c):
    """Update joint distribution of bottleneck variable and x.

    pm: marginal distribution P(M) - vector
    px: marginal distribution P(X) - vector
    pmx_c: conditional distribution P(M|X) - matrix
    """

    
    pmx_j = np.zeros(np.shape(pmx_c)) #P(M,J) to be returned
    for mi in range(len(pm)):
        for xi in range(len(px)):
            pmx_j[mi,xi] = pmx_c[mi,xi]*px[xi]
    return pmx_j

@jit
def p_ym_c(pm,px,py,pyx_c,pmx_c):
    """Update conditional distribution of bottleneck variable given y.
    
    pm: Marginal distribution p(M)
    px: marginal distribution p(X)
    pyx_c: conditional distribution p(Y|X)
    pmx_c: conditional distribution p(M|X)
    """
    pym_c = np.zeros(np.shape(pyx_c))
    for yi in range(len(py)):
        for mi in range(len(pm)):
            for xi in range(len(px)):
                pym_c[yi,mi] += (1./pm[mi])*pyx_c[yi,xi]*pmx_c[mi,xi]*px[xi]
    return pym_c


@jit
def p_m(pmx_c,px):
    """Update marginal distribution of bottleneck variable.

    pmx_c: conditional distribution p(M|X)
    px: marginal distribution p(X)
    """
    pm = np.zeros(len(px))
    for mi in range(len(pm)):
        for xi in range(len(px)):
            pm[mi] += pmx_c[mi,xi]*px[xi]
    return pm


