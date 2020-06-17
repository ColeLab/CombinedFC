#compute the partial correlation matrix and only keep the statistically significant coefficients
#according to a two-sided null hypothesis test at the chosen alpha.
#significant partial correlation coefficients represent edges in the connectivity network.

import numpy as np
from .parCorrRegression import *
from .parCorrInvCov import *
from .parCorrGlasso import *
from .Zcutoff import *
from .fisherZTrans import *


def partialCorrelationSig(dataset, alpha = 0.01, method = 'inverseCovariance'):
    '''
    INPUT:
        dataset : in dimension [nDatapoints x nNodes]
        alpha : cutoff for the significance decision. Default is 0.01
        method : a string, 'regression','inverseCovariance', or 'glasso'
    OUTPUT:
        M : a connectivity network of significant partial correlations
    '''
    
    D = dataset
    nNodes = D.shape[1] #number of variables in the dataset
    nDatapoints = D.shape[0] #number of datapoints
        
    if method == 'regression':   
        #compute the partial correlation matrix using the regression approach
        Mparcorr = parCorrRegression(D)
        condSetSize = nNodes-2
        
    elif method == 'inverseCovariance':
        #compute the partial correlation matrix using the inverse covariance approach
        Mparcorr = parCorrInvCov(D)
        condSetSize = nNodes-2
        
    elif method == 'glasso':
        #compute the partial correlation matrix using graphical lasso
        Mreg, Mmod,condSetSizeMat = parCorrGlasso(D, kfolds=10)
        Mparcorr = Mmod
        #here, each partial correlation has a different conditioning set size
        condSetSize = condSetSizeMat 
        
    #two-sided null hypothesis test of partial correlation = 0.
    #get the Zalpha cutoff
    Zalpha = Zcutoff(alpha = alpha, kind = 'two-sided')    
    #Fisher z-transformation of the partial correlation matrix for the null hypothesis Ho: parcorr = 0
    #partial correlation of 2 nodes conditions on all the rest of the nodes,
    #so the size of the conditioning set is the number of total nodes minus 2.
    Fz = fisherZTrans(Mparcorr, nDatapoints=nDatapoints, Ho=0, condSetSize=condSetSize)
    #threshold the par.corr. matrix using the significant decision abs(Fz) >= Zalpha
    M = np.multiply(Mparcorr, abs(Fz) >= Zalpha) + 0 #+0 is to avoid -0 in the output   
       
    
    return M