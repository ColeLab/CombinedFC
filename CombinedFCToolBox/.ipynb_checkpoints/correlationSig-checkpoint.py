#Compute correlation r and use an equivalence test to determine if r is equal to zero,
#Lakens, D. (2017), Equivalence Test... 8(4), 355-362.
#or a null hypothesis test to determine if r is different from zero,
#a significant r different from zero represents an edges in the connectivity network.

import numpy as np
from .Zcutoff import *
from .fisherZTrans import *

def correlationSig(dataset, 
                   alpha = 0.01, 
                   lower_bound = -0.1, 
                   upper_bound = +0.1, 
                   equivalenceTest = False):
    '''
    INPUT:
        dataset : a dataset with dimension [nDatapoints x nNodes]
        alpha : significance level for the equivalence test or for the two-sided null hypothesis test. Default = 0.01
        lower_bound : a negative bound for the minimum r of interest in the equivalence test. Default = -0.1
        upper_bound : a positive bound for the minimum r of interest in the equivalence test. Default = +0.1
            *note, the bounds are not required to be symmetric.
        equivalenceTest = if True perform the equivalence test, otherwise perform the two-sided null hypothesis test
    OUTPUT:
        M : a matrix such that:
            if equivalenceTest = True, M contains r values judged to be zero, according to the equivalence test.
            if equivalenceTest = False, M contains r values judged different from zero, as per the null hypothesis test. 
    '''
    
    D = dataset
    nNodes = D.shape[1] #number of variables
    nDatapoints = D.shape[0] #number of datapoints
    
    #compute the full correlation matrix
    Mcorr = np.corrcoef(D, rowvar=False)
        
    if equivalenceTest == False: 
        #get the Zalpha cutoff
        Zalpha = Zcutoff(alpha = alpha, kind = 'two-sided')    
        #make the correlation matrix diagonal equal to zero to avoid problems with Fisher z-transformation
        np.fill_diagonal(Mcorr,0)
        #Fisher z-transformation of the correlation matrix for the null hypothesis of Ho: r = 0
        Fz = fisherZTrans(Mcorr, nDatapoints = nDatapoints, Ho = 0)
        #threshold the correlation matrix judging significance if: abs(Fz) >= +Zalpha
        #(this is equivalent to test Fz >= +Zalpha or Fz <= -Zalpha)
        M = np.multiply(Mcorr, abs(Fz) >= Zalpha)+0 #+0 is to avoid -0 in the output
        
    
    elif equivalenceTest == True:    
        #the equivalence test is formed by two one-sided tests
        #Zalpha cutoffs for two one-sided test at a chosen alpha
        #upper bound 
        Zalpha_u = Zcutoff(alpha = alpha, kind = 'one-sided-left')
        #lower bound
        Zalpha_l = Zcutoff(alpha = alpha, kind = 'one-sided-right')
        #make the correlation matrix diagonal equal to zero to avoid problems with Fisher z-transformation
        np.fill_diagonal(Mcorr,0)     
        #Fisher z-transform using Ho: r = upper_bound, Ha: r < upper_bound
        Fz_u = fisherZTrans(Mcorr, nDatapoints = nDatapoints, Ho = upper_bound)
        #and Fisher z-transform using Ho: r = lower_bound, Ha: r > lower_bound
        Fz_l = fisherZTrans(Mcorr, nDatapoints = nDatapoints, Ho = lower_bound)
        #Fz_u = -Fz_l, expressing it as two variables is just for clarity of exposition.
        
        #threshold the correlation matrix judging significantly equal to zero if:
        #Fz_u <= Zalpha_u & Fz_l >= Zalpha_l
        #if both inequalities hold then we judge Fz ~ 0, and thus r ~ 0
        #M contains the correlation values r that were judged close to zero in the equivalence test
        M = np.multiply(Mcorr, np.multiply(Fz_u <= Zalpha_u, Fz_l >= Zalpha_l))+0
        
    return M