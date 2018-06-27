import numpy as np
from numba import jit

@jit
def p_dist(x,numunique=None,window=1):
    """
    Take a sequence and return a marginal distribution

    x = array of observations
    numunique = number of unique values x can take
    window = moving window size of the symbols to take into account

    return an empirical probability distribution of length numunique**window
    """
    if numunique==None:
        unique = np.unique(x).size
    numxwords = numunique**window #number of total possible words given the number of unique symbols and the window size
    px = np.zeros(numunique**window)
    aux_x_base = numunique**np.arange(window)[::-1]
    for i in range(len(x)-window+1):
        px[x[i:i+window] @ aux_x_base] += 1
    return px/px.sum()


@jit
def p_joint(x1,x2,numuniquex1=None,numuniquex2=None,windowx1=1,windowx2=1):
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
    if numuniquex1==None:
        numuniquex1 = np.unique(x1).size
    if numuniquex2==None:
        numuniquex2 = np.unique(x2).size
    numwordsx1 = numuniquex1**windowx1
    numwordsx2 = numuniquex2**windowx2
    aux_base_x1 = numuniquex1**np.arange(windowx1)[::-1]
    aux_base_x2 = numuniquex2**np.arange(windowx2)[::-1]
    px1x2 = np.zeros((numwordsx1,numwordsx2)) #matrix of size numwordsx,numwordsy with for the joint probability distribution
    for i in range(len(x1)-windowx1):
        x1i = (x1[i:i+windowx1] @ aux_base_x1).astype(np.int)
        x2i = (x2[i:i+windowx2] @ aux_base_x2).astype(np.int)
        px1x2[x1i,x2i] += 1
    return px1x2/px1x2.sum()

@jit
def p_cond(px1,px2,px1x2_j):
    """Compute conditional distribution p(x1|x2)

    px2 = probability distribution for the item on which things are conditioned

    px1x2 = joint probability distribution p(x1,x2)

    return a matrix with the same dimensions as px1x2 with
    probabilities of x1 conditioned on x2

    """
    px1x2_c = np.zeros(np.shape(px1x2_j))
    for x1i in range(len(px1)):
        for x2i in range(len(px2)):
            if px2[x2i] > 0:
                px1x2_c[x1i,x2i] = px1x2_j[x1i,x2i]/px2[x2i]
    return px1x2_c/px1x2_c.sum(axis=0) # make sure it's normalized


@jit
def mi_x1x2_c(px1,px2,px1x2_c):
    """Compute the MI between two probability distributions x1 and x2
    using their respective marginals and conditional distribution
    """

    mi = 0.
    for x2i in range(len(px2)):
        for x1i in range(len(px1)):
            if px1x2_c[x1i,x2i] > 0 and px1[x1i] > 0:
                mi += px2[x2i]*px1x2_c[x1i,x2i]*log2(px1x2_c[x1i,x2i]/px1[x1i])
    return mi    
