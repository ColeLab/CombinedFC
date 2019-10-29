#Group correlation analysis using a one sample two-sided t-test,
#to determine if the group mean correlations are significantly different from zero

import numpy as np
import CombinedFCToolBox as cfc
from scipy import stats

def groupCorrSig(groupData, alpha = 0.01):
    '''
    INPUT:
        groupData : a list containing the nSubjects individual datasets in shape (nDatapoints x nNodes)
        alpha : significance level for the one sample two-sided t-test for the null Ho : mean(Fz_xy) = 0 
    OUTPUT:
        gSigCorr : a matrix (nNodes x nNodes) with significant group mean correlation values. 
                   Non-significant correlations are zeroed.
    '''
    
    #get the number of subjects for the group analysis
    nSubjects = len(groupData)
    #get the number of nodes in the dataset. All the subjects have the same number of nodes
    nNodes = groupData[0].shape[1]
    
    #allocate memory for the individual results
    gCorr = np.zeros((nNodes,nNodes,nSubjects))
    Fz_gCorr = np.zeros((nNodes,nNodes,nSubjects))
    
    #compute individual subject correlations and Fisher z-transforms
    for subj in range(nSubjects):
        #compute the correlation matrix
        gCorr[:,:,subj] = np.corrcoef(groupData[subj],rowvar=False)
        #zero the diagonal values, they are 1 otherwise
        np.fill_diagonal(gCorr[:,:,subj],0)
        #Transform the correlation matrix for each subject with Fisher z-transform  
        # z = (atanh(r)-atanh(r_Ho))*sqrt(N-Z-3) ~ N(0,1)
        Fz_gCorr[:,:,subj] = cfc.fisherZTrans(gCorr[:,:,subj], 
                                          nDatapoints = groupData[subj].shape[0], 
                                          Ho=0, 
                                          condSetSize=0)
        
    #Compute group level significance of an edge correlation:                       
    #Use a t-test for the null hypothesis Ho : mean(Fz_xy) = 0, where the mean is across all subjects
    #and Fz_xy, correspond to the Fisher z-transform of the correlation between node x and node y.
    #Given that the matrices are symmetric, just do it for the upper triangular and copy
    #the result in the lower triangular

    #allocate memory for the group significant mean correlation
    gSigCorr = np.zeros((nNodes,nNodes))
    #one t-test for each edge in the network, ie. for each pair of nodes x and y.
    for x in range(nNodes-1):
        for y in range(x+1,nNodes):
            #pvalue for a 1-sample two-sided t-test with null Ho: mean_sample = 0 (popmean = 0)
            #the sample is the Fisher z-transform correlations of an edge(x,y) across all the subjects
            pval = stats.ttest_1samp(Fz_gCorr[x,y,:],popmean = 0)[1]
            #if the mean is significantly different from zero, 
            #add the corresponding mean correlation
            #to the group connectivity matrix: gSigCorr
            if pval < alpha:
                gSigCorr[x,y] = gSigCorr[y,x] = np.mean(gCorr[x,y,:]) 
                
                
    return gSigCorr