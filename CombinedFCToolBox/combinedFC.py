#combinedFC: start with partial correlation and then remove spurious edges from conditioning on colliders,
#by checking if the corresponding correlation is zero.
#Checking for zero correlation can be made with an equivalence test or with a two-sided null hypothesis test.

import CombinedFCToolBox as cfc

def combinedFC(dataset,  
               methodCondAsso = 'partialCorrelation',
               methodParcorr='inverseCovariance',
               alphaCondAsso = 0.01,
               methodAsso = 'correlation',
               alphaAsso = 0.01,
               equivalenceTestAsso = False,
               lower_bound = -0.1, 
               upper_bound = +0.1):
    '''
    INPUT:
        dataset : in dimension [nDatapoints x nNodes]
        methodCondAsso : a string "partialCorrelation" or "multipleRegression" for the conditional association 
                         first step
        methodParcorr : a string if partial correlation is chosen, "inverseCovariance", "regression", "glasso"
        alphaCondAsso : alpha significance cutoff for the conditional association. Default = 0.01
        methodAsso : a string "correlation" or "simpleRegression" for the unconditional association
        alphaAsso : alpha significance cutoff for the unconditional association. Defaul = 0.01
        equivalenceTestAsso : if True perform the equivalence test, otherwise perform the two-sided null 
                              hypothesis test
        lower_bound : a negative bound for the minimum r of interest in the equivalence test. Default = -0.1
        upper_bound : a positive bound for the minimum r of interest in the equivalence test. Default = +0.1
            *note, the bounds are not required to be symmetric.
    OUTPUT:
        M: a connectivity matrix with significant partial correlation coefficients after
            removing possible spurious edges from conditioning on colliders, using a zero correlation check.
    '''
    
    #first step: evaluate full conditional associations
    if methodCondAsso == 'partialCorrelation':
        #partial correlation with a two-sided null hypothesis test for the Ho: parcorr = 0
        #using the method chosen by the user
        Mca = cfc.partialCorrelationSig(dataset, alpha=alphaCondAsso, method=methodParcorr)
    
    if methodCondAsso == 'multipleRegression':
        #multiple regression for each node x on the rest of the nodes in the set
        #with a two-sided t-test for the Ho : beta = 0
        Mca = cfc.multipleRegressionSig(dataset, alpha=alphaCondAsso, sigTest=True)
            

    nNodes = dataset.shape[1]
    #second step
    #start with the conditional association matrix, and then make the collider check using correlation
    M = Mca.copy() 

    if methodAsso == 'correlation' and equivalenceTestAsso == True:
        #correlation with the equivalence test for r = 0
        Mcorr = cfc.correlationSig(dataset, alpha=alphaAsso, lower_bound=lower_bound, 
                            upper_bound=upper_bound, equivalenceTest=True)
        #test if two nodes have a significant partial correlation but a zero correlation 
        #this will be evidence of a spurious edge from conditioning on a collider
        for x in range(nNodes-1):
            for y in range(x+1,nNodes):
                 if Mca[x,y] != 0 and Mcorr[x,y] != 0:
                    M[x,y] = M[y,x] = 0 #remove the edge from the connectivity network
    
     
    elif methodAsso == 'correlation' and equivalenceTestAsso == False:
        #correlation with a two-sided null hypothesis test for the null hypothesis Ho: r = 0
        Mcorr = cfc.correlationSig(dataset, alpha=alphaAsso, equivalenceTest=False)
        #test if two nodes have a significant partial correlation but a not significant correlation 
        #this will be evidence of a spurious edge from conditioning on a collider
        for x in range(nNodes-1):
            for y in range(x+1,nNodes):
                if Mca[x,y] != 0 and Mcorr[x,y] == 0:
                    M[x,y] = M[y,x] = 0 #remove the edge from the connectivity network
                    
    elif methodAsso == 'simpleRegression':
        #simple regression for each pair of nodes that have a significant conditional association
        for x in range(nNodes-1):
            for y in range(x+1,nNodes):
                #do both sides, regression coefficients are not symmetric
                if Mca[x,y] != 0:
                    b = cfc.simpleRegressionSig(dataset[:,x],dataset[:,y],alpha=alphaAsso,sigTest=True)
                    if b == 0:
                        M[x,y] = 0 #remove the edge from the connectivity network
                if Mca[y,x] != 0:
                    b = cfc.simpleRegressionSig(dataset[:,y],dataset[:,x],alpha=alphaAsso,sigTest=True)
                    if b == 0:
                        M[y,x] = 0
                
    
    
    return M