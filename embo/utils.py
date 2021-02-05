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

import numpy as np
from scipy.special import entr, rel_entr

def p_joint(x1, x2, windowx1=1, windowx2=1):
    """
    Compute the joint distribution between two data series
    
    x1 = first array (X, or "past" when doing past-future bottleneck)
    x2 = second array (Y, or "future")
    windowx1 = moving window size to consider for the x1 array. As x1 is typically used for the past when doing a past-future bottleneck, this time window is taken backwards, e.g. a window of size 2 means considering {x_{t-1},x_{t}} as a vector-valued sample.
    windowx2 = moving window size to consider for the x2 array. As x2 is typically used for the future when doing a past-future bottleneck, this time window is taken forwards, e.g. a window of size 2 means considering {x_{t},x_{t+1}} as a vector-valued sample.
    
    return a matrix of the joint probability p(x1,x2)
    """
    x1_unique, x1 = np.unique(x1, return_inverse=True)
    x2_unique, x2 = np.unique(x2, return_inverse=True)
    assert(len(x1)==len(x2))
    numsamples = len(x1)
    numuniquex1 = x1_unique.size
    numuniquex2 = x2_unique.size
    numwordsx1 = numuniquex1**windowx1
    numwordsx2 = numuniquex2**windowx2
    aux_base_x1 = numuniquex1**np.arange(windowx1)[::-1]
    aux_base_x2 = numuniquex2**np.arange(windowx2)[::-1]
    px1x2 = np.zeros((numwordsx1,numwordsx2)) # matrix of size numwordsx,numwordsy for the joint probability distribution
    for i in range(windowx1-1, numsamples-(windowx2-1)):
        x1i = np.inner(x1[i-windowx1+1:i+1], aux_base_x1).astype(np.int)
        x2i = np.inner(x2[i:i+windowx2], aux_base_x2).astype(np.int)
        px1x2[x1i,x2i] += 1
    return px1x2/px1x2.sum()

def entropy(p, axis=0):
    """Compute entropy (in bits) of the given probability distribution.

    Arguments:
       p -- distribution for which the entropy is to be computed. This will be normalized to sum to 1 along the axis of interest.
       axis -- axins along which to compute the entropy (default: 0)
    """
    p = np.asarray(p)
    p = 1.0*p / np.sum(p, axis=axis, keepdims=True)
    return np.sum(entr(p), axis=axis)/np.log(2)

def kl_divergence(p, q, axis=0):
    """Compute KL divergence (in bits) between p and q, DKL(P||Q)."""
    p = np.asarray(p)
    p = 1.0*p / np.sum(p, axis=axis, keepdims=True)
    q = np.asarray(q)
    q = 1.0*q / np.sum(q, axis=axis, keepdims=True)
    return np.sum(rel_entr(p,q), axis=axis)/np.log(2)
    
def mi_x1x2_c(px1, px2, px1x2_c):
    """Compute the MI between two probability distributions x1 and x2
    using their respective marginals and conditional distribution
    """
    marginal_entropy = entropy(px1)
    conditional_entropy = px2 @ entropy(px1x2_c, axis=0)
    return marginal_entropy - conditional_entropy

def compute_upper_bound(IX, IY, betas=None):
    """Remove all points in an IB sequence that would make it nonmonotonic.

    This is a post-processing step that is needed after computing an
    IB sequence (defined as a sequence of (IX, IY) pairs),
    to remove the random fluctuations in the result induced by the AB
    algorithm getting stuck in local minima.

    Parameters
    ----------
    IX : array
        I(M:X) values
    IY : array 
        I(M:Y) values
    betas : array (default None)
        beta values from the IB computation

    Returns
    -------
    array (n x 2)
        (I(M:X), I(M:Y)) coordinates of the IB bound after ensuring monotonic progression (with increasing beta) in both coordinates.
    array (n)
        The beta values corresponding to the points of the upper bound.

    """
    points = np.vstack((IX,IY)).T
    selected_idxs = [0]

    for idx in range(1,points.shape[0]):
        if points[idx,0]>points[selected_idxs[-1],0] and points[idx,1]>=points[selected_idxs[-1],1]:
            selected_idxs.append(idx)
            
    upper_bound = points[selected_idxs, :]

    if betas is None:        
        return upper_bound
    else:
        return upper_bound, betas[selected_idxs]

