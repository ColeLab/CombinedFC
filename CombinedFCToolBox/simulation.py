#define parameters to set a simulation and repeat it a number of times using the repetitionsParallel.py function
#each repetition instantiate a new case of the random graphical model and corresponding data

import multiprocessing as mp
import numpy as np
import CombinedFCToolBox as cfc

def simulation(repetitions=10,
               model='ErdosRenyi',
               nNodes = [50,200,400], 
               nDatapoints = [1200],
               edgeDensity='fixed',
               mean_coeff = 0.8, 
               std_coeff = 1,
               dataType='pseudoEmpirical',
               methodCondAsso = 'partialCorrelation',
               methodParcorr='inverseCovariance',
               methodAsso = 'correlation',
               alphaSig = [0.01],
               cfcEquivalenceTest = False,
               equiv_bounds = [0.1]):
    '''
    INPUT:
        repetitions : number of times the model will be instantiated under different W coefficient values.
        model : string defining the type of network model, 'ErdosRenyi' or 'StaticPowerLaw'
        nNodes : number of nodes for the true model, enter as array [200] or [50,200,400] when analyzing this parameter
        nDatapoints : number of datapoints for the true model, enter as array [1200] or [250,400,1200]
        edgeDensity : if 'fixed' choose the middle value of the 3 edge density parameters of the sensitivity test, 
                      if 'profile' run the 3 edge density parameters
        mean_coeff : mean for the normal distribution from where W coefficients will be sampled.
        std_coeff :  standard deviation for the normal distribution from where W coefficients will be sampled.
        dataType : 'pseudoEmpirical' or 'synthetic': see simulateData.py function for definitions.
        methodCondAsso : a string 'partialCorrelation' or 'multipleRegression' for the conditional association
        methodParcorr : a string 'inverseCovariance' or 'regression', method to compute partial correlation if chosen 
        methodAsso : a string 'correlation' or 'simpleRegression' for the unconditional association
        alphaSig : level of significance for the hypothesis tests, enter as array [0.01] or [0.001,0.01,0.05].
        cfcEquivalenceTest: True, if the equivalence test is used for combinedFC, 
                            otherwise the two-sided null hypothesis test of significance is used.
        equiv_bounds : bounds for the equivalence test, an array [0.2] to test lower bound -0.2 and upper bound +0.2

    OUTPUT:
        precision and recall for correlation, partial correlation and combinedFC, for each repetition of the simulation,
        and for the three values of the parameter that was analyzed, which is the array with 3 elements.
        The result are saved in the Accuracy_xxx arrays, one for each of the methods. See below.
    '''


    if model == 'StaticPowerLaw':
        if edgeDensity == 'fixed':
            parameters = [0.03*2]
        elif edgeDensity == 'profile':    
            parameters = [0.03*2,0.063*2,0.133*2] # % of total possible edges, used to compute the number of edges.
    
    
    elif model == 'ErdosRenyi':
        if edgeDensity == 'fixed':
            parameters = [0.05*2] # % of total possible edges, used this to compute number of edges
        elif edgeDensity == 'profile':    
            parameters = [0.05*2,0.10*2,0.20*2] 
    
    #allocate memory for the accuracy measures and the number of edges in the true graph
    #the last dimension is to save values of precision and recall
    Accuracy_corr = np.zeros((repetitions,len(nNodes),len(parameters),len(nDatapoints),len(equiv_bounds),len(alphaSig),2))
    Accuracy_parcorr = np.zeros((repetitions,len(nNodes),len(parameters),len(nDatapoints),len(equiv_bounds),len(alphaSig),2))
    Accuracy_combfc = np.zeros((repetitions,len(nNodes),len(parameters),len(nDatapoints),len(equiv_bounds),len(alphaSig),2))
    nEdges = np.zeros((repetitions,len(nNodes),len(parameters),len(nDatapoints),len(equiv_bounds),len(alphaSig)))

    for param in range(len(parameters)):
        
        for node in range(len(nNodes)):
        
            for sample in range(len(nDatapoints)):

                for bounds in range(len(equiv_bounds)):
                    
                    for cutoff in range(len(alphaSig)):
                        
                        #initialize the pool for the parallel computations
                        #useful info: www.machinelearningplus.com/python/parallel-processing-python/
                        pool = mp.Pool(mp.cpu_count())
                        #run repetitions in parallel
                        resultsRep = [pool.apply(cfc.repetitionsParallel,
                                                 args=(model,
                                                       nNodes,nDatapoints,
                                                       parameters,
                                                       mean_coeff,std_coeff,
                                                       dataType,
                                                       methodCondAsso,
                                                       methodParcorr,
                                                       methodAsso,
                                                       alphaSig,
                                                       cfcEquivalenceTest,
                                                       equiv_bounds,
                                                       param,node,sample,bounds,cutoff)) 
                                      for rep in range(repetitions)]
                        #close the pool
                        pool.close()
                        
                        #assign the results of the parallelization to the allocated matrices
                        for rep in range(repetitions):
                            Accuracy_corr[rep,node,param,sample,bounds,cutoff,:] = resultsRep[rep][0]
                            Accuracy_parcorr[rep,node,param,sample,bounds,cutoff,:] = resultsRep[rep][1]
                            Accuracy_combfc[rep,node,param,sample,bounds,cutoff,:] = resultsRep[rep][2]
                            nEdges[rep,node,param,sample,bounds,cutoff] = resultsRep[rep][3]

    
    return Accuracy_corr, Accuracy_parcorr, Accuracy_combfc, nEdges

