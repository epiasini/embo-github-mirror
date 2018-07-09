from __future__ import division
import numpy as np

from functools import reduce
from numba import jit
from scipy.stats import entropy
from scipy.spatial import ConvexHull

from .utils import p_joint, mi_x1x2_c


def empirical_bottleneck(x,y,numuniquex=0,numuniquey=0,**kw):
    """ Compute an IB curve for two empirical sequences x and y"""
    
    # Marginal, joint and conditional distributions required to calculate the IB
    pxy_j = p_joint(x,y)
    px = pxy_j.sum(axis=1)
    py = pxy_j.sum(axis=0)
    pyx_c = pxy_j.T / px
    #Calculate the information bottleneck for different values of beta
    i_p,i_f,beta = IB(px,py,pyx_c,**kw)
    # Return array of ipasts and ifutures for array of different values of beta - mi should correspond to the saturation point
    mi = mi_x1x2_c(py, px, pyx_c)
    return i_p,i_f,beta,mi,entropy(px,base=2),entropy(py,base=2)

def IB(px,py,pyx_c,maxbeta=5,numbeta=30,iterations=100):
    """Compute an Information Bottleneck curve

    px: marginal probability distribution for the past
    py: marginal distribution for the future
    maxbeta: the maximum value of beta to use to compute the curve
    iterations: number of iterations to use to for the curve to converge for each value of beta
    
    return vectors of ipast and ifuture (ips and ifs respectively) for
    different values of beta (bs)

    """
    pm_size = px.size
    bs = np.linspace(0.01,maxbeta,numbeta) #value of beta
    ips = np.zeros(bs.size)
    ifs = np.zeros(bs.size)
    for bi in range(bs.size):
        # initialize distribution for bottleneck variable
        pm = np.random.rand(pm_size)+1
        pm /= pm.sum()
        pym_c = np.random.rand(py.size,pm.size)+1 # Starting point for the algorithm
        pym_c /= pym_c.sum(axis=0)
        # iterate the BA algorithm
        for i in range(iterations):
            pmx_c = p_mx_c(pm,px,py,pyx_c,pym_c,bs[bi])
            pm = p_m(pmx_c,px)
            pym_c = p_ym_c(pm,px,py,pyx_c,pmx_c)
            if i>0 and np.allclose(pmx_c,pmx_c_old):
                # if the x->m mapping is not updating any more, we're at convergence and we can stop
                break
            pmx_c_old = pmx_c

        ips[bi] = mi_x1x2_c(pm,px,pmx_c)
        ifs[bi] = mi_x1x2_c(py,pm,pym_c)
    # restrict the returned values to those that, at each value of
    # beta, actually increase (for Ipast) and do not decrease (for
    # Ifuture) the information with respect to the previous value of
    # beta. This is to avoid confounds from cases where the AB
    # algorithm gets stuck in a local minimum.
    ub, betas = compute_upper_bound(ips, ifs, bs)
    return np.squeeze(ub[:,0]), np.squeeze(ub[:,1]), betas

def compute_upper_bound(IX, IY, betas=None):
    """Extract the upper part of the convex hull of an IB sequence.

    This is a post-processing step that is needed after computing an
    IB sequence (defined as a sequence of (IX, IY) pairs),
    to remove the random fluctuations in the result induced by the AB
    algorithm getting stuck in local minima.

    Parameters
    ----------
    IX : array
        I(X) values
    IY : array 
        I(Y) values
    betas : array (default None)
        beta values from the IB computation

    Returns
    -------
    array (n x 2)
        (I(X), I(Y)) coordinates of the upper part of the convex hull
        defined by the input points.
    array (n)
        The beta values corresponding to the points of the upper bound.

    """
    points = np.vstack((IX,IY)).T
    selected_idxs = [0]

    for idx in range(1,points.shape[0]):
        if points[idx,0]>points[selected_idxs[-1],0] and points[idx,1]>=points[selected_idxs[-1],1]:
            selected_idxs.append(idx)
            
    upper_bound = points[selected_idxs,:]

    if betas is None:        
        return upper_bound
    else:
        return upper_bound, betas[selected_idxs]


@jit
def p_mx_c(pm,px,py,pyx_c,pym_c,beta):
    """Update conditional distribution of bottleneck random variable given x.

    pm: marginal distribution p(M) - vector
    px: marginal distribution p(X) - vector
    py: marginal distribution p(Y) - vector
    pyx_c: conditional distribution p(Y|X) - matrix
    pym_c: conditional distribution p(Y|M) - matrix
    """
    
    pmx_c = np.zeros((pm.size,px.size)) # P(M|X) matrix to be returned
    for mi in range(pm.size):
        for xi in range(px.size):
            pmx_c[mi,xi] = pm[mi] * np.exp(-beta * entropy(pyx_c[:,xi], pym_c[:,mi], base=2))
    return pmx_c/pmx_c.sum(axis=0) #Normalize 


@jit
def p_ym_c(pm,px,py,pyx_c,pmx_c):
    """Update conditional distribution of bottleneck variable given y.
    
    pm: Marginal distribution p(M)
    px: marginal distribution p(X)
    pyx_c: conditional distribution p(Y|X)
    pmx_c: conditional distribution p(M|X)
    """
    pym_c = np.zeros((py.size,pm.size))
    for yi in range(py.size):
        for mi in range(pm.size):
            for xi in range(px.size):
                pym_c[yi,mi] += (1./pm[mi])*pyx_c[yi,xi]*pmx_c[mi,xi]*px[xi]
    return pym_c


@jit
def p_m(pmx_c,px):
    """Update marginal distribution of bottleneck variable.

    pmx_c: conditional distribution p(M|X)
    px: marginal distribution p(X)
    """
    pm = np.zeros(pmx_c.shape[0])
    for mi in range(pm.size):
        for xi in range(px.size):
            pm[mi] += pmx_c[mi,xi]*px[xi]
    return pm


