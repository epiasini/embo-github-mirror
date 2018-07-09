from __future__ import division

import numpy as np
from scipy.stats import entropy

def p_joint(x1,x2,numuniquex1=0,numuniquex2=0,windowx1=1,windowx2=1):
    """
    Compute the joint distribution between two data series
    
    x1 = first array
    x2 = second array
    numuniquex1 = number of unique symbols in the x1 array
    numuniquex2 = number of unique symbols in the x2 array
    windowx1 = moving window size to consider for the x1 array
    windowx2 = moving window size to consider for the x2 array
    
    return a matrix of the joint probability p(x1,x2)
    """
    if numuniquex1==0:
        numuniquex1 = np.unique(x1).size
    if numuniquex2==0:
        numuniquex2 = np.unique(x2).size
    numwordsx1 = numuniquex1**windowx1
    numwordsx2 = numuniquex2**windowx2
    aux_base_x1 = numuniquex1**np.arange(windowx1)[::-1]
    aux_base_x2 = numuniquex2**np.arange(windowx2)[::-1]
    px1x2 = np.zeros((numwordsx1,numwordsx2)) #matrix of size numwordsx,numwordsy with for the joint probability distribution
    for i in range(len(x1)-windowx1):
        x1i = np.inner(x1[i:i+windowx1], aux_base_x1).astype(np.int)
        x2i = np.inner(x2[i:i+windowx2], aux_base_x2).astype(np.int)
        px1x2[x1i,x2i] += 1
    return px1x2/px1x2.sum()

def mi_x1x2_c(px1,px2,px1x2_c):
    """Compute the MI between two probability distributions x1 and x2
    using their respective marginals and conditional distribution
    """
    marginal_entropy = entropy(px1, base=2)
    conditional_entropy = 0.
    for x2i in range(px2.size):
        conditional_entropy += px2[x2i] * entropy(px1x2_c[:,x2i], base=2)
    return marginal_entropy - conditional_entropy

