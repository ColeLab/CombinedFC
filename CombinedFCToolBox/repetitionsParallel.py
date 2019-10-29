#Use this function to be able to run in parallel the repetitions of the simulation.
#This function 1) instantiates the graphical model, 2) generates data, 
#3) computes correlation, partial correlation, combinedFC, 4) gets precision and recall

import numpy as np
import CombinedFCToolBox as cfc

def repetitionsParallel(model,
                        nNodes,
                        nDatapoints,
                        parameters,
                        mean_coeff,
                        std_coeff,
                        dataType,
                        methodCondAsso,
                        methodParcorr,
                        methodAsso,
                        alphaSig,
                        cfcEquivalenceTest,
                        equiv_bounds,
                        param,node,sample,bounds,cutoff):
            
        
            
    check = 0
    while check == 0: 
        
        #generate a networks according to the chosen graphical model
        if model == 'StaticPowerLaw' or model == 'ErdosRenyi':
            totalPosEdges = nNodes[node] * (nNodes[node]-1)/2 #total number of possible edges
            edgeDensity = round(totalPosEdges*parameters[param])
            C = cfc.graphModel(model=model,edgeDensity=edgeDensity, nNodes=nNodes[node]) 

        #Simulate data using network model C
        if dataType == 'pseudoEmpirical':
            X = cfc.simulateData.pseudoEmpiricalData(C, mean_coeff=mean_coeff, std_coeff=std_coeff,
                                                     nDatapoints=nDatapoints[sample])
        if dataType == 'synthetic':
            X = cfc.simulateData.syntheticData(C, mean_coeff=mean_coeff, std_coeff=std_coeff,
                                                     nDatapoints=nDatapoints[sample],
                                                      distribution='Gaussian')

        #compute combinedFC and the chosen methods by themselves
        Mcfc = cfc.combinedFC(dataset=X, 
                              methodCondAsso=methodCondAsso,
                              methodParcorr=methodParcorr,
                              alphaCondAsso=alphaSig[cutoff],
                              methodAsso=methodAsso,
                              alphaAsso=alphaSig[cutoff],
                              equivalenceTestAsso=cfcEquivalenceTest, 
                              lower_bound = -equiv_bounds[bounds],
                              upper_bound = +equiv_bounds[bounds])
        
        if methodAsso == 'correlation':
            Mc = cfc.correlationSig(dataset=X, alpha=alphaSig[cutoff], equivalenceTest=False)
        elif methodAsso == 'simpleRegression':
            Mc = np.zeros((nNodes[node],nNodes[node]))
            for x in range(nNodes[node]-1):
                for y in range(x+1,nNodes[node]):
                    Mc[x,y] = cfc.simpleRegressionSig(X[:,x],X[:,y],alpha=alphaSig[cutoff],sigTest=True)
                    Mc[y,x] = cfc.simpleRegressionSig(X[:,y],X[:,x],alpha=alphaSig[cutoff],sigTest=True)
        
        
        if methodCondAsso == 'partialCorrelation':
            Mpc = cfc.partialCorrelationSig(dataset=X, alpha=alphaSig[cutoff], method=methodParcorr)
        elif methodCondAsso == 'multipleRegression':
            Mpc = cfc.multipleRegressionSig(dataset=X, alpha=alphaSig[cutoff], sigTest=True)
            
            
        #If the inverse covariance approach produces a nan result repeat the simulation
        if np.any(np.isnan(Mpc)):
            check = 0
        #if not, return the result and break the while-loop to continue
        else:
            check = 1
    
    #get precision and recall for each of the three methods tested
    PrRe_Mc = np.zeros((1,2))
    PrRe_Mc[0,0]   = cfc.precision(inferred_model = Mc,true_model = C)
    PrRe_Mc[0,1]   = cfc.recall(inferred_model = Mc,true_model = C)
    
    PrRe_Mpc = np.zeros((1,2))
    PrRe_Mpc[0,0]  = cfc.precision(inferred_model = Mpc, true_model = C)
    PrRe_Mpc[0,1]  = cfc.recall(inferred_model = Mpc, true_model = C)

    PrRe_Mcfc = np.zeros((1,2))
    PrRe_Mcfc[0,0] = cfc.precision(inferred_model = Mcfc, true_model = C)
    PrRe_Mcfc[0,1] = cfc.recall(inferred_model = Mcfc, true_model = C)
    
    #number of edges in the true graphical model
    nEdges = np.sum(np.triu(np.maximum(C, C.T),1))
    
    return [PrRe_Mc,PrRe_Mpc,PrRe_Mcfc,nEdges]