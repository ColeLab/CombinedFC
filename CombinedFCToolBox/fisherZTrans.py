#Fisher z-transformation Fz, for the correlation/partial correlation coefficient r
#The statistic Fz is approximately distributed as a standard normal ~ N(mean=0,std.dev=1)
#the function works for single values of matrices.

import numpy as np

def fisherZTrans(r, nDatapoints, Ho=0, condSetSize=0):
    '''
    INPUT:
        r : correlation(or partial correlation) coefficient or matrix of coefficients
        Ho : value for the null hypothesis of r, default Ho: r = 0
        nDatapoints : sample size
        condSetSize : the size of the conditioning set of the partial correlation for the Fisher z transform.
            more info:  https://en.wikipedia.org/wiki/Partial_correlation#As_conditional_independence_test
            For correlation the condSetSize = 0, for partial correlation condSetSize > 0.
    OUTPUT:
        Fz: a single Fisher z-transform value or matrix of values
    '''
    
    Fz = np.multiply(np.subtract(np.arctanh(r),np.arctanh(Ho)), np.sqrt(nDatapoints-condSetSize-3))
    
    return Fz