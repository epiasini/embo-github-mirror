import unittest
import numpy as np

import embo


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
