import unittest
import numpy as np

from embo.embo import empirical_bottleneck


class TestSimpleSequences(unittest.TestCase):
    def setUp(self):
        pass

    def test_binary_sequence(self):
        # Fake data sequence
        x = np.array([0,0,0,1,0,1,0,1,0,1]*300)
        y = np.array([1,0,1,0,1,0,1,0,1,0]*300)

        # IB bound for different values of beta
        i_p,i_f,beta,mi = empirical_bottleneck(x,y)

        np.testing.assert_allclose(i_f[-1],mi,rtol=1e-3)
