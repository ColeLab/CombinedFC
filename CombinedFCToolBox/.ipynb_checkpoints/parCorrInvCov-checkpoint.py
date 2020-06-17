#compute the partial correlation using the inverse covariance approach.
#The partial correlation matrix is the negative of the off-diagonal elements of the inverse covariance,
#divided by the squared root of the corresponding diagonal elements.     
#https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
#in this approach, for two nodes X and Y, the partial correlation conditions on all the nodes except X and Y.

import numpy as np
from scipy import linalg

def parCorrInvCov(dataset):
    '''
    INPUT:
        dataset : a dataset with dimension [nDatapoints x nNodes]
    OUTPUT:
        M : matrix of partial correlation coefficients
    '''
    D = dataset
    #compute the covariance matrix of the dataset and invert it. This is known as the precision matrix.
    #use the (Moore-Penrose) pseudo-inverse of a matrix.
    invCovM = linalg.pinv(np.cov(D,rowvar=False))
    #transform the precision matrix into partial correlation coefficients
    denom = np.atleast_2d(1. / np.sqrt(np.diag(invCovM)))
    M = -invCovM * denom * denom.T
    #make the diagonal zero.
    np.fill_diagonal(M,0)

    return M