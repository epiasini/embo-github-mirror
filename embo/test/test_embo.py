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

def test_origin(x, y, maxbeta=10):
    """Check that the IB bound starts at (0,0) for small beta"""
    eb = embo.EmpiricalBottleneck(x, y, maxbeta=maxbeta)
    i_x = eb.get_ix()
    i_y = eb.get_iy()
    np.testing.assert_allclose((i_x[0], i_y[0]), (0,0), rtol=1e-7, atol=1e-9)


def test_asymptote(x, y, maxbeta=10):
    """Check that the IB bound saturates at (H(x),MI(X:Y)) for large beta.
    
    Note that both H(X) and MI(X,Y) are computed using the functions
    defined within EMBO.

    """
    eb = embo.EmpiricalBottleneck(x, y, maxbeta=maxbeta)
    i_x = eb.get_ix()
    i_y = eb.get_iy()
    mi = eb.get_saturation_point()
    hx, hy = eb.get_entropies()

    np.testing.assert_allclose((i_x[-1], i_y[-1]), (hx, mi), rtol=1e-7, atol=1e-9)


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


class TestBinarySequence(unittest.TestCase):
    def setUp(self):
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
        self.xcorr = np.zeros_like(self.xycorr, dtype=np.float)
        self.ycorr = np.zeros_like(self.xycorr, dtype=np.float)
        self.xcorr[np.logical_or(self.xycorr==2, self.xycorr==3)] = 1
        self.ycorr[np.logical_or(self.xycorr==1, self.xycorr==3)] = 1

        self.x_with_nan = np.copy(self.x).astype(np.float)
        self.x_with_nan[-1] = np.NaN
        self.x_with_inf = np.copy(self.x).astype(np.float)
        self.x_with_inf[-1] = np.Inf
        self.empty = np.array([])

    def test_origin(self):
        """Check beta->0 limit for fixed binary sequence"""
        test_origin(self.x, self.y)
        
    def test_origin_stochastic(self):
        """Check beta->0 limit for stochastic binary sequence"""
        eb = embo.EmpiricalBottleneck(self.xcorr, self.ycorr)
        test_origin(self.xcorr, self.ycorr)
        
    def test_asymptote(self):
        """Check beta->infinity limit for fixed binary sequence"""
        test_asymptote(self.x, self.y)

    def test_asymptote_stochastic(self):
        """Check beta->infinity limit for stochastic binary sequence"""
        test_asymptote(self.xcorr, self.ycorr, maxbeta=15)

    def test_empty_arrays(self):
        """Check that computing the IB on a pair of empty arrays just returns zero"""
        eb = embo.EmpiricalBottleneck(self.empty, self.empty)
        i_x, i_y, _, mixy, hx, hy = eb.get_empirical_bottleneck(return_entropies=True)
        np.testing.assert_equal(i_x, np.zeros(1))
        np.testing.assert_equal(i_y, np.zeros(1))
        np.testing.assert_equal(mixy, np.zeros(1))
        np.testing.assert_equal(hx, np.zeros(1))
        np.testing.assert_equal(hy, np.zeros(1))

    def test_array_with_nan(self):
        """Check that an error is raised in presence of NaNs"""
        with self.assertRaises(
                ValueError,
                msg="The observation data contains NaNs or Infs."):
            embo.EmpiricalBottleneck(self.x_with_nan, self.y)

    def test_array_with_inf(self):
        """Check that an error is raised in presence of infinities"""
        with self.assertRaises(
                ValueError,
                msg="The observation data contains NaNs or Infs."):
            embo.EmpiricalBottleneck(self.x_with_inf, self.y)


class TestUpperBound(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[0, 0], [1, 1], [2, 0], [3, 3], [3, 4], [2, 5], [4, 6], [2, 7], [3, 8]])
        self.betas = np.arange(self.a.shape[0])
        self.true_idxs = np.array([0, 1, 3, 6], dtype=np.int)

    def test_upper_bound(self):
        """Check extraction of upper bound in IB space"""
        u = embo.compute_upper_bound(self.a[:, 0], self.a[:, 1])
        np.testing.assert_array_equal(u, self.a[self.true_idxs, :])

    def test_betas(self):
        """Check extraction of beta values related to upper bound in IB space"""
        u, betas = embo.compute_upper_bound(self.a[:,0], self.a[:,1], self.betas)
        np.testing.assert_array_equal(betas, self.betas[self.true_idxs])


class TestArbitraryAlphabet(unittest.TestCase):
    def setUp(self):
        # Fake data sequence
        self.x = np.array([0, 0, 0, 2, 0, 2, 0, 2, 0, 2])
        self.y = np.array([3.5, 0, 3.5, 0, 3.5, 0, 3.5, 0, 3.5, 0])

    def test_origin(self):
        """Check beta->0 limit for sequence with arbitrary alphabet"""
        test_origin(self.x, self.y)
        
    def test_asymptote(self):
        """Check beta->infinity limit for sequence with arbitrary alphabet"""
        test_asymptote(self.x, self.y)


if __name__ == "__main__":
    unittest.main(verbosity=2)
