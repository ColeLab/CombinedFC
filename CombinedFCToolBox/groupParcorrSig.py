#Group partial correlation analysis using a one sample two-sided t-test,
#determine if the group mean partial correlations are significantly different from zero

import numpy as np
import CombinedFCToolBox as cfc
from scipy import stats

def groupParcorrSig(groupData, method = 'inverseCovariance', alpha = 0.01):
    '''
    INPUT:
        groupData : a list containing the nSubjects individual datasets in shape (nDatapoints x nNodes)
        method : 'inverseCovariance' or 'regression' to compute partial correlations
        alpha : significance level for the one sample two-sided t-test for the null Ho : mean(Fz_xy) = 0 
    OUTPUT:
        gSigParcorr : a matrix (nNodes x nNodes) with significant group mean partial correlation values. 
                   Non-significant partial correlations are zeroed.
    '''
    
    #get the number of subjects for the group analysis
    nSubjects = len(groupData)
    #get the number of nodes in the dataset. All the subjects must have the same number of nodes
    nNodes = groupData[0].shape[1]
    
    #allocate memory for the individual results
    gParcorr = np.zeros((nNodes,nNodes,nSubjects))
    Fz_gParcorr = np.zeros((nNodes,nNodes,nSubjects))
    
    #compute individual subject partial correlations and Fisher z-transforms
    for subj in range(nSubjects):
        #compute the partial correlation matrix using the selected method
        if method == 'inverseCovariance':
            gParcorr[:,:,subj] = cfc.parCorrInvCov(groupData[subj])
        elif method == 'regression':
            gParcorr[:,:,subj] = cfc.parCorrRegression(groupData[subj])
        #Transform the partial correlation matrix for each subject with Fisher's transform 
        Fz_gParcorr[:,:,subj] = cfc.fisherZTrans(gParcorr[:,:,subj], 
                                             nDatapoints = groupData[subj].shape[0], 
                                             Ho = 0, 
                                             condSetSize = nNodes-2)
    
        
    #Compute group level significance of an edge mean partial correlation:                       
    #Use a t-test for the null hypothesis Ho : mean(Fz_xy) = 0, where the mean is across all subjects
    #and Fz_xy, correspond to the Fisher z-transform of the partial correlation between node x and node y.
    #Given that the matrices are symmetric, just do it for the upper triangular and copy
    #the result in the lower triangular
    
    #allocate memory for the group significant mean partial correlation matrix
    gSigParcorr = np.zeros((nNodes,nNodes))
    #one t-test for each edge in the network, ie. for each pair of nodes x and y.
    for x in range(nNodes-1):
        for y in range(x+1,nNodes):
            #pvalue for a one-sample two-sided t-test with null Ho: mean_sample = 0 (popmean = 0)
            #the sample is the Fisher z-transform correlations of an edge(x,y) across all the subjects
            pval = stats.ttest_1samp(Fz_gParcorr[x,y,:],popmean=0)[1]
            #if the mean is significantly different from zero, 
            #add the corresponding mean partial correlation coefficient
            #to the group connectivity matrix: gSigParcorr
            if pval < alpha:
                gSigParcorr[x,y] = gSigParcorr[y,x] = np.mean(gParcorr[x,y,:])
     
    
    return gSigParcorr