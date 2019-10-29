#Group combinedFC analysis

import numpy as np
import CombinedFCToolBox as cfc
from scipy import stats

def groupCombinedFC(groupData, 
                    alpha = 0.01, 
                    methodParcorr ='inverseCovariance', 
                    equivalenceTest = False,
                    lower_bound = -0.2, 
                    upper_bound = +0.2,
                    alphaEqTest = 0.01):
    '''
    INPUT:
        groupData : a list containing the nSubjects individual datasets in shape (nDatapoints x nNodes)
        alpha : significance level for the one sample two-sided t-test for the null Ho : mean(Fz_xy) = 0 
        methodParcorr : 'inverseCovariance' or 'regression' for the partial correlation computation
        equivalenceTest : if True perform the equivalence test, otherwise perform the one sample two-sided t-test
        lower_bound : a negative bound for the minimum correlation of interest in the equivalence test
        upper_bound : a positive bound for the minimum correlation of interest in the equivalence test
            *note, the bounds are not required to be symmetric.
        alphaEqTest : an alpha level for the equivalence test     
    OUTPUT:
        gSigCombFC : a matrix (nNodes x nNodes) with combinedFC group result.           
    '''
    
    #get the number of subjects for the group analysis
    nSubjects = len(groupData)
    #get the number of nodes in the dataset. All the subjects have the same number of nodes
    nNodes = groupData[0].shape[1]
    
    #First step: 
    #compute group significant mean partial correlation
    gSigParcorr = cfc.groupParcorrSig(groupData, method = methodParcorr, alpha = alpha)
    
    #Second step: 
    #compute correlations and Fisher z-transforms for each subject
    #allocate memory for the individual results
    gCorr = np.zeros((nNodes,nNodes,nSubjects))
    Fz_gCorr = np.zeros((nNodes,nNodes,nSubjects))
    #loop through each subject
    for subj in range(nSubjects):
        #compute the correlation matrix
        gCorr[:,:,subj] = np.corrcoef(groupData[subj],rowvar=False)
        #zero the diagonal values, they are 1 otherwise
        np.fill_diagonal(gCorr[:,:,subj],0)
        #Transform the correlation matrix for each subject with Fisher z-transform  
        #z = (atanh(r)-atanh(r_Ho))*sqrt(N-Z-3) ~ N(0,1)
        Fz_gCorr[:,:,subj] = cfc.fisherZTrans(gCorr[:,:,subj], 
                                          nDatapoints = groupData[subj].shape[0], 
                                          Ho=0, 
                                          condSetSize=0)
    
    #Third step:
    #determine group zero correlations using non-signficant correlations or equivalence tests (ie. collider check)
    
    #allocate memory
    gZeroCorr = np.ones((nNodes,nNodes))
    
    if equivalenceTest == False:
        #combinedFC using group non-significant correlations
        #compute a one-sample two-sided t-test for each pair of nodes x and y
        for x in range(nNodes-1):
            for y in range(x+1,nNodes):
                pval = stats.ttest_1samp(Fz_gCorr[x,y,:],popmean=0)[1]
                #here we want the values that are NOT significantly different from zero
                #thus we need those with pvalue >= alpha significance.
                #In other words, we DO NOT want to reject Ho: mean(Fz_xy) = 0
                if pval >= alpha:
                    #results are symmetric
                    gZeroCorr[x,y] = gZeroCorr[y,x] = 0
                    
    elif equivalenceTest == True:
        #combinedFC using an equivalence tests for a one-sample t-test (see Lakens, 2017)
        #Transform the minimum correlation of interests, "lower_bound" and "upper_bound" to a Fisher z-statistic
        l_b = cfc.fisherZTrans(lower_bound, nDatapoints = groupData[subj].shape[0], Ho=0, condSetSize=0)
        u_b = cfc.fisherZTrans(upper_bound, nDatapoints = groupData[subj].shape[0], Ho=0, condSetSize=0)
        #value that the group mean M is tested against in the one-sample t-test Ho: M = Mu
        Mu = 0 
        #Equivalence test for each edge X-Y
        for x in range(nNodes-1):
            for y in range(x+1,nNodes):
                M = np.mean(Fz_gCorr[x,y,:],axis=0)
                sd = np.std(Fz_gCorr[x,y,:],ddof=1)
                #compute the t-statistics for the upper and lower bounds
                t_l = (M - Mu - l_b)/(sd/np.sqrt(nSubjects))
                t_u = (M - Mu - u_b)/(sd/np.sqrt(nSubjects))
                #p-value for the lower bound t-statistic for a one-sided-right test
                pval_l = 1 - stats.t.cdf(t_l, df = nSubjects-1)
                #p-value for the upper bound t-statistic for a one-sided-left test
                pval_u = stats.t.cdf(t_u, df = nSubjects-1)
                #If both p-values are < alpha, then the mean Fisher z-statistic is judge zero.
                if pval_l < alphaEqTest and pval_u < alphaEqTest:
                    gZeroCorr[x,y] = gZeroCorr[y,x] = 0

        
        
    #Fourth step:
    #the following multiplication makes 0 the significant partial correlations that
    #have a zero correlation, since this could be evidence of an spurious 
    #edge from conditioning on a collider.
    #The remaining significant partial correlations are not altered.
    gSigCombFC = np.multiply(gSigParcorr,gZeroCorr)
       
    return gSigCombFC