#function to compute precision, which is a measure of accuracy to infer presence/absence of edges.
#see definition below.
#https://en.wikipedia.org/wiki/Precision_and_recall

import numpy as np

def precision(inferred_model, true_model):
    '''
    INPUT:
        inferred_model: an inferred connectivity network (may or not may be binarized)
        true_model: the true connectivity network (may or not may be binarized)
    OUTPUT:
        precision: a value between 0 and 1
    '''

    #Binarized the inferred connectivity network and the true connectivity network
    inferred_model = (inferred_model != 0).astype(int)
    true_model = (true_model != 0).astype(int)
    #symmetrize the matrices, 
    #since we are only interested in presence/absence of edges and not their direction
    inferred_model = np.maximum(inferred_model, inferred_model.T)
    true_model = np.maximum(true_model, true_model.T)
    
    #compute true positives: number of correctly inferred true edges
    Tp = np.dot(inferred_model.flatten(),true_model.flatten())
    #true positives + false positives = number of inferred edges
    Tp_plus_Fp = np.sum(inferred_model.flatten())
    #compute precision = true positives / (true positives + false positives)
    precision = Tp/Tp_plus_Fp
    
    return precision