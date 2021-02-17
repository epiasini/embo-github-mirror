# Copyright (C) 2020,2021 Eugenio Piasini, Alexandre Filipowicz,
# Jonathan Levine.
#
# This file is part of embo.
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

import unittest
import numpy as np

import embo

from scipy.stats import entropy as spentropy

def test_origin(x, y, alpha=1):
    """Check that the IB bound starts at (0,0) for small beta"""
    eb = embo.InformationBottleneck(x, y, alpha=alpha)
    i_x = eb.get_ix()
    i_y = eb.get_iy()
    np.testing.assert_allclose((i_x[0], i_y[0]), (0,0), rtol=1e-7, atol=1e-6)


def test_asymptote(x, y, maxbeta=30, alpha=1):
    """Check that the IB/DIB bound saturates at MI(X:Y) for large beta.

    This should be true with the default setting that the cardinality
    of M is the same as that of X.
    
    Note that the reference value of MI(X,Y) is computed using embo's
    internal facility for this.

    """
    eb = embo.InformationBottleneck(x, y, maxbeta=maxbeta, alpha=alpha)
    i_y = eb.get_iy()
    mi = eb.get_saturation_point()
    np.testing.assert_allclose(i_y[-1], mi, rtol=1e-7, atol=1e-6)


class TestEntropyFunctions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng()
        self.p = np.array([0, 0.3, 0.7])
        self.q = np.array([0.1, 0.5, 0.4])
        self.p_unnormalized = 2*self.p
        # define a collection of 50 unnormalized distributions over 10 states
        self.p_multiple = self.rng.random((10,50))
        self.q_multiple = self.rng.random(self.p_multiple.shape)

    def test_entropy_normalized(self):
        """Check entropy function on one normalized distribution"""
        np.testing.assert_array_equal(
            embo.utils.entropy(self.p),
            spentropy(self.p, base=2))

    def test_entropy_unnormalized(self):
        """Check entropy function on one unnormalized distribution"""
        np.testing.assert_array_equal(
            embo.utils.entropy(self.p_unnormalized),
            spentropy(self.p_unnormalized, base=2))

    def test_kl(self):
        """Check KL divergence between two single distributions"""
        np.testing.assert_array_equal(
            embo.utils.kl_divergence(self.p, self.q),
            spentropy(self.p, self.q, base=2))
    
    def test_entropy_multiple(self):
        """Check entropy for multiple distributions"""
        np.testing.assert_array_equal(
            embo.utils.entropy(self.p_multiple),
            spentropy(self.p_multiple, base=2))

    def test_kl_multiple(self):
        """Check KL divergence between multiple distributions"""
        np.testing.assert_array_equal(
            embo.utils.kl_divergence(self.p_multiple, self.q_multiple),
            spentropy(self.p_multiple, self.q_multiple, base=2))


class TestRandomDistribution(unittest.TestCase):
    """Run IB/DIB with random distributions

    Each of the tests here runs on a certain number of random
    distributions, for all combinations of X and Y up to certain
    limits.

    """
    def setUp(self):
        # alpha values for which to run all tests
        self.alphas = np.linspace(0,1,5)
        # maximum cardinality of X and Y
        self.Xmax = 4
        self.Ymax = 4
        # number of random distributions to try for each shape
        self.N = 3
        # random number generator
        self.rng = np.random.default_rng()

    def sample_pxy(self, X, Y, alpha_x=1000, alpha_y=0.1):
        px = self.rng.dirichlet(np.full(X, alpha_x))
        pyx_c = np.zeros((Y,X))
        for xi in range(X):
            pyx_c[:,xi] = self.rng.dirichlet(np.full(Y, alpha_y))
            pyx = pyx_c * px[np.newaxis,:]
        return pyx.T

    def test_random_distributions(self):
        """Run IB/DIB on many random distributions of various shapes and check β→∞ limit"""
        for X in range(2, self.Xmax):
            for Y in range(2, self.Ymax):
                for alpha in self.alphas:
                    with self.subTest(X=X, Y=Y, alpha=alpha):
                        for each in range(self.N):
                            #pxy = self.rng.dirichlet(np.full(X*Y,1/2)).reshape(X,Y)
                            pxy = self.sample_pxy(X,Y)
                            eb = embo.InformationBottleneck(pxy=pxy, maxbeta=20, numbeta=5, alpha=alpha, restarts=10, rtol=1e-5, iterations=1000, ensure_monotonic_bound=False)
                            ix, iy, hm, _, ixy, hx, hy = eb.get_bottleneck(return_entropies=True)
                            self.assertTrue(
                                # note the relatively large
                                # tolerances: they help ignoring
                                # uninteresting cases where I(X:Y) is
                                # very small
                                np.testing.assert_allclose(iy[-1], ixy, rtol=1e-1, atol=1e-1) is None,
                                msg='Check asymptote')


class TestBinarySequence(unittest.TestCase):
    def setUp(self):
        # alpha values for which to run all tests
        self.alphas = np.linspace(0,1,5)
        # Simple data sequence
        self.x = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
        self.y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        # Random data sequence
        self.rng = np.random.default_rng()
        probs = np.array([0.05, 0.45, 0.45, 0.05]) # 0:00 1:01 2:10 3:11
        cumprobs = np.cumsum(probs)
        rand = self.rng.random(1000)
        self.xycorr = np.zeros_like(rand)
        for cp in cumprobs[:-1]:
            self.xycorr[rand>=cp] += 1
        self.xcorr = np.zeros_like(self.xycorr, dtype=float)
        self.ycorr = np.zeros_like(self.xycorr, dtype=float)
        self.xcorr[np.logical_or(self.xycorr==2, self.xycorr==3)] = 1
        self.ycorr[np.logical_or(self.xycorr==1, self.xycorr==3)] = 1

        self.x_with_nan = np.copy(self.x).astype(float)
        self.x_with_nan[-1] = np.NaN
        self.x_with_inf = np.copy(self.x).astype(float)
        self.x_with_inf[-1] = np.Inf
        self.empty = np.array([])

    def test_origin(self):
        """Check β→0 limit for fixed binary sequence"""
        for alpha in self.alphas:
            with self.subTest(alpha=alpha):
                test_origin(self.x, self.y, alpha=alpha)

    def test_origin_stochastic(self):
        """Check β→0 limit for stochastic binary sequence"""
        for alpha in self.alphas:
            with self.subTest(alpha=alpha):
                test_origin(self.xcorr, self.ycorr, alpha=alpha)
        
    def test_asymptote(self):
        """Check β→∞ limit for fixed binary sequence"""
        for alpha in self.alphas:
            with self.subTest(alpha=alpha):
                test_asymptote(self.x, self.y, alpha=alpha)

    def test_asymptote_stochastic(self):
        """Check β→∞ limit for stochastic binary sequence"""
        for alpha in self.alphas:
            with self.subTest(alpha=alpha):
                test_asymptote(self.xcorr, self.ycorr, alpha=alpha)

    def test_empty_arrays(self):
        """Check that computing the IB on a pair of empty arrays raises an exception"""
        for alpha in self.alphas:
            with self.subTest(alpha=alpha):
                 with self.assertRaises(
                        ValueError,
                        msg="If pxy is not specified, x and y can't be empty."):
                     eb = embo.InformationBottleneck(self.empty, self.empty, alpha=alpha)

    def test_array_with_nan(self):
        """Check that an error is raised in presence of NaNs"""
        for alpha in self.alphas:
            with self.subTest(alpha=alpha):
                with self.assertRaises(
                        ValueError,
                        msg="The observation data contains NaNs or Infs."):
                    embo.InformationBottleneck(self.x_with_nan, self.y, alpha=alpha)

    def test_array_with_inf(self):
        """Check that an error is raised in presence of infinities"""
        for alpha in self.alphas:
            with self.subTest(alpha=alpha):
                with self.assertRaises(
                        ValueError,
                        msg="The observation data contains NaNs or Infs."):
                    embo.InformationBottleneck(self.x_with_inf, self.y, alpha=alpha)


class TestUpperBound(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[0, 0], [1, 1], [2, 0], [3, 3], [3, 4], [2, 5], [4, 6], [2, 7], [3, 8]])
        self.betas = np.arange(self.a.shape[0])
        self.true_idxs = np.array([0, 1, 3, 6], dtype=int)

    def test_upper_bound(self):
        """Check extraction of upper bound in IB space"""
        u, _  = embo.compute_upper_bound(self.a[:, 0], self.a[:, 1])
        np.testing.assert_array_equal(u, self.a[self.true_idxs, :])

    def test_betas(self):
        """Check extraction of beta values related to upper bound in IB space"""
        u, idxs = embo.compute_upper_bound(self.a[:,0], self.a[:,1])
        betas = self.betas[idxs]
        np.testing.assert_array_equal(betas, self.betas[self.true_idxs])


class TestArbitraryAlphabet(unittest.TestCase):
    def setUp(self):
        # Fake data sequence
        self.x = np.array([0, 0, 0, 2, 0, 2, 0, 2, 0, 2])
        self.y = np.array([3.5, 0, 3.5, 0, 3.5, 0, 3.5, 0, 3.5, 0])
        # alpha values for which to run all tests
        self.alphas = np.linspace(0,1,5)

    def test_origin(self):
        """Check β→0 limit for sequence with arbitrary alphabet"""
        for alpha in self.alphas:
            with self.subTest(alpha=alpha):
                test_origin(self.x, self.y, alpha=alpha)
        
    def test_asymptote(self):
        """Check β→∞ limit for sequence with arbitrary alphabet"""
        for alpha in self.alphas:
            with self.subTest(alpha=alpha):
                test_asymptote(self.x, self.y, alpha=alpha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
