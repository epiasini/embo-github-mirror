# Copyright (C) 2020,2021 Eugenio Piasini, Alexandre Filipowicz,
# Jonathan Levine.
#
# This file is part of Embo.
#
# Embo is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Embo is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Embo.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import multiprocessing as mp

from .utils import p_joint, mi_x1x2_c, compute_upper_bound, kl_divergence, entropy

np.seterr(divide='ignore', invalid='ignore')

class InformationBottleneck:

    def __init__(self, x=None, y=None, alpha=1, window_size_x=1, window_size_y=1, pxy=None, **kwargs):
        """ Information Bottleneck analysis for an empirical dataset (x,y) or a joint probability mass function pxy.

            Arguments:
            x -- first empirical observation sequence ("past" if doing past-future bottleneck analysis)
            y -- second empirical observation sequence ("future")
            alpha -- generalized bottleneck parameter: alpha=1 is IB and alpha=0 is DIB (deterministic bottleneck)
            window_size_x, window_size_y (int) -- size of the moving windows to be used to compute the IB curve (you typically don't need to worry about this unless you're doing a "past-future bottleneck"-type analysis). The time window on x (which in these cases is typically the "past") is taken backwards, and the time window on y (the "future") is taken forwards. For instance, setting window_size_x=3 and window_size_y=2 will yield the IB curve between (X_{t-2},X_{t-1},X_{t}) and (Y_{t},Y_{t+1}).
            pxy -- joint probability distribution of X and Y. This is a numpy array such that pxy[xi,yi] is the joint probability of the xi-th value of X and the yi-th value of Y. If the array does not sum to one, it will be normalized to ensure that it does. If this argument is passed, the joint estimation step will be skipped; therefore it can only be passed if x and y are None.
            kwargs -- additional keyword arguments to be passed to IB().

        """

        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.alpha = alpha
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y
        self.kwargs_IB = kwargs
        self.results_ready = False
        if (x is not None and not np.all(np.isfinite(self.x))) or\
           (y is not None and not np.all(np.isfinite(self.y))):
            raise ValueError("The observation data contains NaNs or Infs.")
        if pxy is not None and (x is not None or y is not None):
            raise ValueError("It is not possible to specify both pxy and the empirical data x,y.")
        self.pxy_j = pxy
        if self.pxy_j is not None:
            if not np.all(self.pxy_j>=0):
                raise ValueError("Negative values in the specified joint p(X,Y).")
            self.pxy_j /= 1.0*self.pxy_j.sum()
        elif x is None or y is None:
            raise ValueError("Eiter pxy or x and y should be specified.")
        elif len(x)==0 or len(y)==0:
            raise ValueError("If pxy is not specified, x and y can't be empty.")
        self.get_empirical_bottleneck = self.get_bottleneck

    def compute_IB_curve(self):
        """ Compute the IB curve for the joint empirical observations for X and Y. """
        
        # Marginal, joint and conditional distributions required to calculate the IB
        if self.pxy_j is None:
            self.pxy_j = p_joint(self.x, self.y, self.window_size_x, self.window_size_y)

        px = self.pxy_j.sum(axis=1)
        py = self.pxy_j.sum(axis=0)
        pyx_c = self.pxy_j.T / px

        # Calculate the information bottleneck for a range of values of beta
        self.i_x, self.i_y, self.h_m, self.beta, self.mixy, self.hx = self.IB(px, py, pyx_c, self.alpha, **self.kwargs_IB)
        self.hy = entropy(py)

        # set a flag we will use to call this function automatically when needed
        self.results_ready = True
    
    def get_bottleneck(self, return_entropies=False):
        """Return array of I(M:X) and I(M:Y) for array of different values of beta

         Returns:
            i_x -- values of I(M:X) for each value of beta
            i_y -- values of I(M:Y) for each value of beta
            h_m -- values of H(M) for each value of beta
            beta -- values of beta considered
            mixy -- mutual information between X and Y, I(X:Y) (curve saturation point) (only returned if return_entropies is True)
            hx -- entropy of X (only returned if return_entropies is True)
            hy -- entropy of Y (only returned if return_entropies is True)
        """
        if not self.results_ready:
            self.compute_IB_curve()
        
        if return_entropies:
            return self.i_x, self.i_y, self.h_m, self.beta, self.mixy, self.hx, self.hy
        else:
            return self.i_x, self.i_y, self.h_m, self.beta
    
    def get_ix(self):
        if not self.results_ready:
            self.compute_IB_curve()
        return self.i_x

    def get_iy(self):
        if not self.results_ready:
            self.compute_IB_curve()
        return self.i_y

    def get_hm(self):
        if not self.results_ready:
            self.compute_IB_curve()
        return self.h_m

    def get_beta_values(self):
        if not self.results_ready:
            self.compute_IB_curve()
        return self.beta

    def get_saturation_point(self):
        if not self.results_ready:
            self.compute_IB_curve()
        return self.mixy

    def get_entropies(self):
        if not self.results_ready:
            self.compute_IB_curve()
        return self.hx, self.hy

    @classmethod
    def beta_iter(cls, a, b, px, py, pyx_c, pm_size, restarts, iterations, rtol=1e-3, atol=0):
        """Function to run BA algorithm for individual values of beta

        Arguments:
        a -- value of alpha defining the generalized bottleneck (a=1 is IB, a=0 is DIB)
        b -- value of beta on which to run algorithm
        px -- marginal probability distribution for X
        py -- marginal probability distribution for Y
        pyx_c -- conditional distribution p(y|x)
        pm_size -- discrete size of the compression distribution
        restarts -- number of times the optimization procedure should be restarted (for each value of beta) from different random initial conditions
        iterations -- maximum number of iterations to perform
        rtol, atol -- relative and absolute tolerances on the cost function to determine when the algorithm has converged

        Returns:
        list with i_x and i_y, which correspond to I(M:X) and I(M:Y) values for each value of beta
        """
        candidates = []
        for r in range(restarts):

            # initialization
            if pm_size==px.size and pm_size>0:
                # by default pm_size will be the same as px.size. In
                # this case we initialize similarly to Strouse and
                # Schwab 2016: for x_j, most of the mass of the
                # encoder p(m|x) is on m_j, while a smaller part of
                # the mass is distributed randomly over the other
                # values of m.
                pmx_c = np.zeros((pm_size, px.size))
                delta = 0.2 * np.random.rand(1,px.size)
                pmx_c[0,:] = 0.75 - delta
                pmx_c[1:,:] = np.random.rand(pm_size-1, px.size)+1
                pmx_c[1:,:] = (0.25+delta)*pmx_c[1:,:]/(pmx_c[1:,:].sum(axis=0))
                for kx in range(pmx_c.shape[1]):
                    pmx_c[:,kx] = np.roll(pmx_c[:,kx], kx)
                pm = p_m(pmx_c, px)
                pym_c = p_ym_c(pm, px, pyx_c, pmx_c)
            else:
                # legacy initialization scheme
                pm = np.random.rand(pm_size)+1
                pm /= pm.sum()
                pym_c = np.random.rand(py.size, pm.size)+1  # Starting point for the algorithm
                pym_c /= pym_c.sum(axis=0)
                    
            
            # Iterate the BA algorithm
            for i in range(iterations):
                pmx_c, z = p_mx_c(pm, px, pyx_c, pym_c, a, b)
                pm = p_m(pmx_c,px)
                pm, pmx_c, pym_c = drop_unused_dimensions(pm, pmx_c, pym_c)
                pym_c = p_ym_c(pm, px, pyx_c, pmx_c)
                pm, pmx_c, pym_c = drop_unused_dimensions(pm, pmx_c, pym_c)
                # compute cost: H(M)-αH(M|X)-βI(M:Y)
                if a==1:
                    # specialized efficient form for the regular IΒ
                    # (eq. 29 in Tishby 2000; equivalent to the
                    # formula below, but faster)
                    cost = -px @ np.log(z)
                else:
                    # generalized bottleneck
                    cost = entropy(pm) - a * px @ entropy(pmx_c, axis=0) - b * mi_x1x2_c(py, pm, pym_c)
                if i > 0 and np.allclose(cost, cost_old, rtol=rtol, atol=atol):
                    # if the cost function is not changing any more, we're at convergence and we can stop
                    break
                cost_old = cost
            candidates.append({'info_x': mi_x1x2_c(pm, px, pmx_c),
                               'info_y': mi_x1x2_c(py, pm, pym_c),
                               'entropy_m' : entropy(pm),
                               'cost': cost})
        # among the restarts, select the result with minimum cost
        selected_candidate = min(candidates, key=lambda c: c['cost'])
        i_x = selected_candidate['info_x']
        i_y = selected_candidate['info_y']
        h_m = selected_candidate['entropy_m']
        return [i_x, i_y, h_m, b]

    @classmethod
    def IB(cls, px, py, pyx_c, alpha=1, minsize = False, maxbeta=5, minbeta=1e-2, numbeta=30, iterations=100, restarts=3, processes=1, ensure_monotonic_bound='auto', rtol=1e-3):
        """Compute an Information Bottleneck curve

        Arguments:
        px -- marginal probability distribution for X
        py -- marginal probability distribution for Y
        pyx_c -- conditional probability of Y given X
        alpha -- generalized bottleneck parameter: alpha=1 is IB, alpha=0 is DIB
        minsize -- if True, maximum size of compression matches smallest codebook size between x and y
        maxbeta -- the maximum value of beta to use to compute the curve.
        minbeta -- the minimum value of beta to use.
        numbeta -- the number of (equally-spaced) beta values to consider to compute the curve.
        iterations -- number of iterations to use to for the curve to converge for each value of beta
        restarts -- number of times the optimization procedure should be restarted (for each value of beta) from different random initial conditions.
        processes -- number of cpu threads to run in parallel (default = 1)
        ensure_monotonic_bound -- one of 'entropy', 'information', 'auto' or False. This parameter can be used to ensure that the final bound is monotonic in the chosen plane. If 'information', monotonicity is encforced in the IB plane. If 'entropy', monotonicity is enforced in the DIB plane. If 'auto', monotonicity is enforced in the IB plane for alpha=1 and in the DIB plane for alpha==0, but not for intermediate values. If False, monotonicity is not enforced and the original beta sequence is preserved.
        rtol -- relative tolerance parameter for convergence criterion.

        Returns:
        ips -- values of I(M:X) for each beta considered
        ifs -- values of I(M:Y) for each beta considered
        hms -- values of H(M) for each beta considered
        bs -- values of beta considered. Note that this does not necessarily match the set of values initially considered (numbeta equally spaced values between 0.01 and maxbeta). As numerical accuracies can lead to to solutions (I(M:X), I(M:Y)) that would make the IB curve nonmonotonic, any such solution (as well as the corresponding beta value) is discarded before returning the results. See utils.compute_upper_bound() for more details.
        mixy -- mutual information between x and y (curve saturation point)
        hx -- entropy of x (maximum I(M:X) value)
        """
        pm_size = px.size
        if minsize:
            pm_size = min(px.size,py.size)  # Get compression size - smallest size
        
        bs = np.linspace(minbeta, maxbeta, numbeta)  # value of beta

        # Parallel computing of compression for desired beta values
        with mp.Pool(processes=processes) as pool:
            results = [pool.apply_async(cls.beta_iter, args=(alpha, b, px, py, pyx_c, pm_size, restarts, iterations,rtol)) for b in bs]
            results = [p.get() for p in results]
        ips = [x[0] for x in results]
        ifs = [x[1] for x in results]
        hms = [x[2] for x in results]
        bs = [x[3] for x in results]

        # Values of beta may not be sorted appropriately due to out-of
        # order execution if using many processes. So we have so sort
        # the result lists (ix, iy, hm, beta) in ascending beta order.
        ips = [x for _, x in sorted(zip(bs, ips))]
        ifs = [x for _, x in sorted(zip(bs, ifs))]
        hms = [x for _, x in sorted(zip(bs, hms))]
        bs = sorted(bs)

        # restrict the returned values to those that, at each value of
        # beta, actually increase (for I(M:X) or H(M)) and do not
        # decrease (for I(M:Y)) the information/entropy with respect
        # to the previous value of beta. This is to avoid confounds
        # from cases where the AB algorithm gets stuck in a local
        # minimum.
        if ensure_monotonic_bound:
            if ensure_monotonic_bound=='auto':
                if alpha==1:
                    # If alpha ==1 (vanilla IB), we ensure monotonicity in the IB plane
                    ensure_monotonic_bound = 'information'
                elif alpha==0:
                    # If alpha==0, we ensure monotonicity in the DIB plane
                    ensure_monotonic_bound = 'entropy'
            if ensure_monotonic_bound=='information':
                ub, idxs = compute_upper_bound(ips, ifs)
                ips = np.squeeze(ub[:, 0])
                ifs = np.squeeze(ub[:, 1])
                hms = np.array(hms)[idxs]
                bs = np.array(bs)[idxs]
            elif ensure_monotonic_bound=='entropy':
                ub, idxs = compute_upper_bound(hms, ifs)
                hms = np.squeeze(ub[:, 0])
                ifs = np.squeeze(ub[:, 1])
                ips = np.array(ips)[idxs]
                bs = np.array(bs)[idxs]

        # Return saturation point (mixy) and max horizontal axis (hx)
        mixy = mi_x1x2_c(py, px, pyx_c)
        hx = entropy(px)
        return ips, ifs, hms, bs, mixy, hx

def elementwise_l(pm, px, pyx_c, pym_c, alpha, beta):
    """Log-loss function, elementwise.

    This is the generalization of l_β(x,m) in Strouse 2016 to the case
    where 0<α≤1.

    """
    return (np.log(pm[:,np.newaxis]) - beta * kl_divergence(pyx_c[:,np.newaxis,:], pym_c[:,:,np.newaxis]))/alpha

def p_mx_c(pm, px, pyx_c, pym_c, alpha, beta):
    """Update conditional distribution of bottleneck random variable given x.

    Arguments:
    pm -- marginal distribution p(M) - vector
    px -- marginal distribution p(X) - vector
    pyx_c -- conditional distribution p(Y|X) - matrix
    pym_c -- conditional distribution p(Y|M) - matrix
    alpha -- H(M|X) weight: alpha=1 is IB, alpha=0 is DIB
    beta -- I(M:Y) weight

    Returns:
    pmx_c -- conditional distribution p(M|X) (encoder)
    z -- normalizing factor
    """
    if alpha > 0:
        # Generalized Information Bottleneck. Note that vanilla IB
        # corresponds to the alpha=1 case.
        pmx_c = np.exp(elementwise_l(pm, px, pyx_c, pym_c, alpha, beta))
        z = pmx_c.sum(axis=0)
        pmx_c /= z # normalize
    elif alpha==0:
        # Deterministic Information Bottleneck. As per Algorithm 2 in
        # Strouse 2016, we compute the log-loss function for the
        # vanilla IB (α=1), and for each value of x we set it to 1 for
        # the value of t for which it's maximum, and zero otherwise.
        l = elementwise_l(pm, px, pyx_c, pym_c, 1, beta)
        pmx_c = np.zeros_like(l)
        pmx_c[np.argmax(l, axis=0), np.arange(l.shape[1])] = 1
        z = 1
    return pmx_c, z

def p_ym_c(pm, px, pyx_c, pmx_c):
    """Update conditional distribution of bottleneck variable given y.

    Arguments:
    pm -- Marginal distribution p(M)
    px -- marginal distribution p(X)
    pyx_c -- conditional distribution p(Y|X)
    pmx_c -- conditional distribution p(M|X)

    Returns:
    pym_c -- conditional distribution p(Y|M)
    """
    pym = pyx_c*px[np.newaxis,:] @ pmx_c.T
    pym_c = pym / pm[np.newaxis,:]
    return pym_c

def p_m(pmx_c, px):
    """Update marginal distribution of bottleneck variable.

    Arguments:
    pmx_c -- conditional distribution p(M|X)
    px -- marginal distribution p(X)

    Returns:
    pm - marginal distribution of compression p(M)
    """
    return pmx_c @ px

def drop_unused_dimensions(pm, pmx_c, pym_c, eps=0):
    """Remove bottleneck dimensions that are not in use anymore.

    This is in general safe to do as once a dimension is excluded by
    the algorithm, it is never brought back. It is necessary to use
    this function particularly when using the generalized bottleneck,
    where dimensions are discarded very aggressively, and if keeping
    them we would need to put special safeguards in place to avoid
    NaNs and infinities in the Dkl and when computing p(y|m).

    """
    unused_pm = pm<=eps
    unused_pmx_c = np.all(pmx_c<=eps, axis=1)
    unused_pym_c = np.all(pym_c<=eps, axis=0)
    unused = np.any(np.vstack([unused_pm, unused_pmx_c.T, unused_pym_c]), axis=0)
    in_use = ~unused
    return pm[in_use], pmx_c[in_use,:], pym_c[:,in_use]
