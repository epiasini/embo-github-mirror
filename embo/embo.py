from __future__ import division

import numpy as np

from scipy.stats import entropy

from .utils import p_joint, mi_x1x2_c, compute_upper_bound


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


def IB(px,py,pyx_c,maxbeta=5,numbeta=30,iterations=100,restarts=3):
    """Compute an Information Bottleneck curve

    px: marginal probability distribution for the past
    py: marginal distribution for the future
    maxbeta: the maximum value of beta to use to compute the curve
    iterations: number of iterations to use to for the curve to converge for each value of beta
    restarts: number of times the optimization procedure should be restarted (for each value of beta) from different random initial conditions.
    
    return vectors of ipast and ifuture (ips and ifs respectively) for
    different values of beta (bs)

    """
    pm_size = px.size
    bs = np.linspace(0.01,maxbeta,numbeta) #value of beta
    ips = np.zeros(bs.size)
    ifs = np.zeros(bs.size)
    for bi in range(bs.size):
        candidates = []
        for r in range(restarts):
            # initialize distribution for bottleneck variable
            pm = np.random.rand(pm_size)+1
            pm /= pm.sum()
            pym_c = np.random.rand(py.size,pm.size)+1 # Starting point for the algorithm
            pym_c /= pym_c.sum(axis=0)
            # iterate the BA algorithm
            for i in range(iterations):
                pmx_c, z = p_mx_c(pm,px,py,pyx_c,pym_c,bs[bi])
                pm = p_m(pmx_c,px)
                pym_c = p_ym_c(pm,px,py,pyx_c,pmx_c)
                if i>0 and np.allclose(pmx_c,pmx_c_old):
                    # if the x->m mapping is not updating any more, we're at convergence and we can stop
                    break
                pmx_c_old = pmx_c
            candidates.append({'past_info' : mi_x1x2_c(pm, px, pmx_c),
                               'future_info' : mi_x1x2_c(py, pm, pym_c),
                               'functional' : -np.log2(np.inner(z,px))})
        # among the restarts, select the result that gives the minimum
        # value for the functional we're actually minimizing (eq 29 in
        # Tishby et al 2000).
        selected_candidate = min(candidates, key=lambda c: c['functional'])
        ips[bi] = selected_candidate['past_info']
        ifs[bi] = selected_candidate['future_info']
    # restrict the returned values to those that, at each value of
    # beta, actually increase (for Ipast) and do not decrease (for
    # Ifuture) the information with respect to the previous value of
    # beta. This is to avoid confounds from cases where the AB
    # algorithm gets stuck in a local minimum.
    ub, bs = compute_upper_bound(ips, ifs, bs)
    ips = np.squeeze(ub[:,0])
    ifs = np.squeeze(ub[:,1])
    return ips, ifs, bs


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
    z = pmx_c.sum(axis=0)
    pmx_c /= z #Normalize 
    return pmx_c, z


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
