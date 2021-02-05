import unittest
import numpy as np

import embo

from scipy.stats import entropy as spentropy

def test_origin(x, y):
    """Check that the IB bound starts at (0,0) for small beta"""
    eb = embo.EmpiricalBottleneck(x, y)
    i_p = eb.get_ipast()
    i_f = eb.get_ifuture()
    np.testing.assert_allclose((i_p[0], i_f[0]), (0,0), rtol=1e-7, atol=1e-9)


def test_asymptote(x, y):
    """Check that the IB bound saturates at (H(x),MI(X:Y)) for large beta.
    
    Note that both H(X) and MI(X,Y) are computed using the functions
    defined within EMBO.

    """
    eb = embo.EmpiricalBottleneck(x, y, maxbeta=10)
    i_p = eb.get_ipast()
    i_f = eb.get_ifuture()
    mi = eb.get_saturation_point()
    hx, hy = eb.get_entropies()

    np.testing.assert_allclose((i_p[-1], i_f[-1]), (hx, mi), rtol=1e-7)


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
        # Fake data sequence
        self.x = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 1])
        self.y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    def test_origin(self):
        """Check beta->0 limit for binary sequence"""
        test_origin(self.x, self.y)
        
    def test_asymptote(self):
        """Check beta->infinity limit for binary sequence"""
        test_asymptote(self.x, self.y)


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
