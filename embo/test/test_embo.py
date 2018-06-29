import unittest
import numpy as np

from embo.embo import empirical_bottleneck

class TestBinarySequence(unittest.TestCase):
    def setUp(self):
        # Fake data sequence
        self.x = np.array([0,0,0,1,0,1,0,1,0,1]*300)
        self.y = np.array([1,0,1,0,1,0,1,0,1,0]*300)

    def test_origin(self):
        """Check that the IB bound starts at (0,0) for small beta"""
        i_p,i_f,beta,mi,_,_ = empirical_bottleneck(self.x,self.y)
        np.testing.assert_allclose((i_p[0],i_f[0]),(0,0),rtol=1e-7,atol=1e-10)

    def test_asymptote(self):
        """Check that the IB bound saturates at (H(x),MI(X:Y)) for large beta.

        Note that both H(X) and MI(X,Y) are computed using the
        functions defined within EMBO.

        """
        i_p,i_f,beta,mi,hx,hy = empirical_bottleneck(self.x,self.y,maxbeta=10)
        np.testing.assert_allclose((i_p[-1],i_f[-1]),(hx,mi),rtol=1e-5)
