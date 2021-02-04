import numpy as np
import multiprocessing as mp

from .utils import p_joint, mi_x1x2_c, compute_upper_bound, kl_divergence, entropy

np.seterr(divide='ignore', invalid='ignore')

class EmpiricalBottleneck:

    def __init__(self, x, y, window_size_x=1, window_size_y=1, **kwargs):
        """ Information Bottleneck curve for an empirical dataset (X,Y), given as an observation sequence for X and one for Y.

            Arguments:
            x -- first empirical observation sequence ("past" if doing past-future bottleneck analysis)
            y -- second empirical observation sequence ("future")
            window_size_x, window_size_y (int) -- size of the moving windows to be used to compute the IB curve (you typically don't need to worry about this unless you're doing a "past-future bottleneck"-type analysis). The time window on x (which in these cases is typically the "past") is taken backwards, and the time window on y (the "future") is taken forwards. For instance, setting window_size_x=3 and window_size_y=2 will yield the IB curve between (X_{t-2},X_{t-1},X_{t}) and (Y_{t},Y_{t+1}).
            kwargs -- additional keyword arguments to be passed to IB().

        """

        self.x = x
        self.y = y
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y
        self.kwargs_IB = kwargs
        self.results_ready = False

    def compute_IB_curve(self):
        """ Compute the IB curve for the joint empirical observations for X and Y. """
        
        # Marginal, joint and conditional distributions required to calculate the IB
        pxy_j = p_joint(self.x, self.y, self.window_size_x, self.window_size_y)
        px = pxy_j.sum(axis=1)
        py = pxy_j.sum(axis=0)
        pyx_c = pxy_j.T / px

        # Calculate the information bottleneck for a range of values of beta
        self.i_x, self.i_y, self.beta, self.mixy, self.hx = self.IB(px, py, pyx_c, **self.kwargs_IB)
        self.hy = entropy(py)

        # set a flag we will use to call this function automatically when needed
        self.results_ready = True

    def get_empirical_bottleneck(self, return_entropies=False):
        """Return array of ipasts and ifutures for array of different values of beta
         mixy should correspond to the saturation point
         Returns:
            i_x -- values of I(M:X) for each value of beta
            i_y -- values of I(M:Y) for each value of beta
            beta -- values of beta considered
            mixy -- mutual information between X and Y, I(X:Y) (curve saturation point) (only returned if return_entropies is True)
            hx -- entropy of X (only returned if return_entropies is True)
            hy -- entropy of Y (only returned if return_entropies is True)
        """
        if not self.results_ready:
            self.compute_IB_curve()
        
        if return_entropies:
            return self.i_x, self.i_y, self.beta, self.mixy, self.hx, self.hy
        else:
            return self.i_x, self.i_y, self.beta
    
    def get_ipast(self):
        if not self.results_ready:
            self.compute_IB_curve()
        return self.i_x

    def get_ifuture(self):
        if not self.results_ready:
            self.compute_IB_curve()
        return self.i_y

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
    def beta_iter(cls, b, px, py, pyx_c, pm_size, restarts, iterations):
        """Function to run BA algorithm for individual values of beta

        Arguments:
        b -- value of beta on which to run algorithm
        px -- marginal probability distribution for X ("past")
        py -- marginal probability distribution for Y ("future")
        pyx_c -- conditional distribution p(y|x)
        pm_size -- discrete size of the compression distribution
        restarts -- number of times the optimization procedure should be restarted (for each value of beta) from different random initial conditions
        iterations -- maximum number of iterations to use until convergence

        Returns:
        list with i_x and i_y, which correspond to I(M:X) and I(M:Y) values for each value of beta
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
        i_x = selected_candidate['past_info']
        i_y = selected_candidate['future_info']
        return [i_x, i_y, b]

    @classmethod
    def IB(cls, px, py, pyx_c, minsize = False, maxbeta=5, numbeta=30, iterations=100, restarts=3, processes=1):
        """Compute an Information Bottleneck curve

        Arguments:
        px -- marginal probability distribution for X
        py -- marginal probability distribution for Y
        pyx_c -- conditional probability of Y given X
        minsize -- if True, maximum size of compression matches smallest codebook size between x and y
        maxbeta -- the maximum value of beta to use to compute the curve. Minimum is 0.01.
        numbeta -- the number of (equally-spaced) beta values to consider to compute the curve.
        iterations -- number of iterations to use to for the curve to converge for each value of beta
        restarts -- number of times the optimization procedure should be restarted (for each value of beta) from different random initial conditions.
        processes -- number of cpu threads to run in parallel (default = 1)

        Returns:
        ips -- values of I(M:X) for each beta considered
        ifs -- values of I(M:Y) for each beta considered
        bs -- values of beta considered. Note that this does not necessarily match the set of values initially considered (numbeta equally spaced values between 0.01 and maxbeta). As numerical accuracies can lead to to solutions (I(M:X), I(M:Y)) that would make the IB curve nonmonotonic, any such solution (as well as the corresponding beta value) is discarded before returning the results. See utils.compute_upper_bound() for more details.
        mixy -- mutual information between x and y (curve saturation point)
        hx -- entropy of x (maximum ipast value)
        """
        pm_size = px.size
        if minsize:
            pm_size = min(px.size,py.size)  # Get compression size - smallest size
        
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
        hx = entropy(px)
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
        pmx_c = pm[:,np.newaxis] * np.exp(-beta * kl_divergence(pyx_c[:,np.newaxis,:], pym_c[:,:,np.newaxis]))
        z = pmx_c.sum(axis=0)
        pmx_c /= z # normalize
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
        return pyx_c*px @ pmx_c.T/pm

    @staticmethod
    def p_m(pmx_c, px):
        """Update marginal distribution of bottleneck variable.

        Arguments:
        pmx_c -- conditional distribution p(M|X)
        px -- marginal distribution p(X)

        Returns:
        pm - marginal distribution of compression p(M)
        """
        return pmx_c @ px
