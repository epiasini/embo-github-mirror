import numpy as np
import multiprocessing as mp

from scipy.stats import entropy
from .utils import p_joint, mi_x1x2_c, compute_upper_bound

np.seterr(divide='ignore', invalid='ignore')


class EmpiricalBottleneck:

    def __init__(self, x, y, window_size_x=1, window_size_y=1, **kwargs):
        """ Compute an IB curve for two empirical sequences x and y

            Arguments:
            x -- first empirical sequence ("past")
            y -- second empirical sequence ("future")
            return_entropies (bool) -- whether to return the marginal entropies of x and y
            kwargs -- additional keyword arguments to be passed to IB().

        """
        # Marginal, joint and conditional distributions required to calculate the IB
        pxy_j = p_joint(x, y, window_size_x, window_size_y)
        px = pxy_j.sum(axis=1)
        py = pxy_j.sum(axis=0)
        pyx_c = pxy_j.T / px

        # Calculate the information bottleneck for a range of values of beta
        self.i_x, self.i_y, self.beta, self.mixy, self.hx = self.IB(px, py, pyx_c, **kwargs)
        self.hy = entropy(py, base=2)

    def get_empirical_bottleneck(self):
        """Return array of ipasts and ifutures for array of different values of beta
         mixy should correspond to the saturation point
         Returns:
            i_p -- values of ipast for each value of beta
            i_f -- values of ifuture for each value of beta
            beta -- values of beta considered
            mixy -- mutual information between x and y (curve saturation point) (only returned if return_entropies is True)
            hx -- entropy of x (only returned if return_entropies is True)
            hy -- entropy of y (only returned if return_entropies is True)
        """
        return self.i_x, self.i_y, self.beta, self.mixy, self.hx, self.hy
    
    def get_ipast(self):
        return self.i_x

    def get_ifuture(self):
        return self.i_y

    def get_beta_values(self):
        return self.beta

    def get_saturation_point(self):
        return self.mixy

    def get_entropies(self):
        return self.hx, self.hy

    @classmethod
    def beta_iter(cls, b, px, py, pyx_c, pm_size, restarts, iterations):
        """Function to run BA algorithm for individual values of beta

        Arguments:
        b -- value of beta on which to run algorithm
        px -- marginal probability distribution for the past (x)
        py -- marginal probability distribution for the future (y)
        pyx_c -- conditional distribution p(y|x)
        pm_size -- discrete size of the compression distribution
        restarts -- number of times the optimization procedure should be restarted (for each value of beta) from different random initial conditions
        iterations -- maximum number of iterations to use until convergence

        Returns:
        list with i_p and i_f, which correspond to ipast and ifuture values for each value of beta
        """
        candidates = []
        for r in range(restarts):

            # Initialize distribution for bottleneck variable
            pm = np.random.rand(pm_size)+1
            pm /= pm.sum()
            pym_c = np.random.rand(py.size, pm.size)+1  # Starting point for the algorithm
            pym_c /= pym_c.sum(axis=0)

            # Iterate the BA algorithm
            for i in range(iterations):
                pmx_c, z = cls.p_mx_c(pm, px, py, pyx_c, pym_c, b)
                pm = cls.p_m(pmx_c,px)
                pym_c = cls.p_ym_c(pm, px, py, pyx_c, pmx_c)
                if i > 0 and np.allclose(pmx_c, pmx_c_old, rtol=1e-3, atol=1e-3):
                    # if the x->m mapping is not updating any more, we're at convergence and we can stop
                    break
                pmx_c_old = pmx_c
            candidates.append({'past_info': mi_x1x2_c(pm, px, pmx_c),
                               'future_info': mi_x1x2_c(py, pm, pym_c),
                               'functional': -np.log2(np.inner(z, px))})
        # among the restarts, select the result that gives the minimum
        # value for the functional we're actually minimizing (eq 29 in
        # Tishby et al 2000).
        selected_candidate = min(candidates, key=lambda c: c['functional'])
        i_p = selected_candidate['past_info']
        i_f = selected_candidate['future_info']
        return [i_p, i_f, b]

    @classmethod
    def IB(cls, px, py, pyx_c, maxbeta=5, numbeta=30, iterations=100, restarts=3, processes=1):
        """Compute an Information Bottleneck curve

        Arguments:
        px -- marginal probability distribution for the past
        py -- marginal distribution for the future
        maxbeta -- the maximum value of beta to use to compute the curve
        iterations -- number of iterations to use to for the curve to converge for each value of beta
        restarts -- number of times the optimization procedure should be restarted (for each value of beta) from different random initial conditions.
        processes -- number of cpu threads to run in parallel (default = 1)

        Returns:
        ips -- values of ipast for each beta considered
        ifs -- values of ifuture for each beta considered
        bs -- values of beta considered
        mixy -- mutual information between x and y (curve saturation point)
        hx -- entropy of x (maximum ipast value)
        """

        pm_size = px.size  # Get compression size
        bs = np.linspace(0.01, maxbeta, numbeta)  # value of beta

        # Parallel computing of compression for desired beta values
        pool = mp.Pool(processes=processes)
        results = [pool.apply_async(cls.beta_iter, args=(b, px, py, pyx_c, pm_size, restarts, iterations,)) for b in bs]
        pool.close()
        results = [p.get() for p in results]
        ips = [x[0] for x in results]
        ifs = [x[1] for x in results]

        # Values of beta may not be sorted appropriately.
        # code below sorts ipast and ifuture according to their corresponding value of beta, and in correct order
        b_s = [x[2] for x in results]
        ips = [x for _, x in sorted(zip(b_s, ips))]
        ifs = [x for _, x in sorted(zip(b_s, ifs))]

        # restrict the returned values to those that, at each value of
        # beta, actually increase (for Ipast) and do not decrease (for
        # Ifuture) the information with respect to the previous value of
        # beta. This is to avoid confounds from cases where the AB
        # algorithm gets stuck in a local minimum.
        ub, bs = compute_upper_bound(ips, ifs, bs)
        ips = np.squeeze(ub[:, 0])
        ifs = np.squeeze(ub[:, 1])

        # Return saturation point (mixy) and max horizontal axis (hx)
        mixy = mi_x1x2_c(py, px, pyx_c)
        hx = entropy(px, base=2)
        return ips, ifs, bs, mixy, hx

    @staticmethod
    def p_mx_c(pm, px, py, pyx_c, pym_c, beta):
        """Update conditional distribution of bottleneck random variable given x.

        Arguments:
        pm -- marginal distribution p(M) - vector
        px -- marginal distribution p(X) - vector
        py -- marginal distribution p(Y) - vector
        pyx_c -- conditional distribution p(Y|X) - matrix
        pym_c -- conditional distribution p(Y|M) - matrix

        Returns:
        pmx_c -- conditional distribution p(M|X)
        z -- normalizing factor
        """

        pmx_c = np.zeros((pm.size, px.size))  # P(M|X) matrix to be returned
        for mi in range(pm.size):
            for xi in range(px.size):
                pmx_c[mi,xi] = pm[mi] * np.exp(-beta * entropy(pyx_c[:, xi], pym_c[:, mi], base=2))
        z = pmx_c.sum(axis=0)
        pmx_c /= z  # Normalize
        return pmx_c, z

    @staticmethod
    def p_ym_c(pm, px, py, pyx_c, pmx_c):
        """Update conditional distribution of bottleneck variable given y.

        Arguments:
        pm -- Marginal distribution p(M)
        px -- marginal distribution p(X)
        pyx_c -- conditional distribution p(Y|X)
        pmx_c -- conditional distribution p(M|X)

        Returns:
        pym_c -- conditional distribution p(Y|M)
        """
        pym_c = np.zeros((py.size, pm.size))
        for yi in range(py.size):
            for mi in range(pm.size):
                for xi in range(px.size):
                    pym_c[yi, mi] += (1./pm[mi])*pyx_c[yi, xi]*pmx_c[mi, xi]*px[xi]
        return pym_c

    @staticmethod
    def p_m(pmx_c, px):
        """Update marginal distribution of bottleneck variable.

        Arguments:
        pmx_c -- conditional distribution p(M|X)
        px -- marginal distribution p(X)

        Returns:
        pm - marginal distribution of compression p(M)
        """
        pm = np.zeros(pmx_c.shape[0])
        for mi in range(pm.size):
            for xi in range(px.size):
                pm[mi] += pmx_c[mi, xi]*px[xi]
        return pm
