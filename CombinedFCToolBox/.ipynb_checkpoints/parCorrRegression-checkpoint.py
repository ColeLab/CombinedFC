#compute the partial correlation using the regression approach.
#https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
#In this approach, for two nodes X and Y, the partial correlation conditions on all the nodes except X and Y.

import numpy as np
from sklearn.linear_model import LinearRegression

def parCorrRegression(dataset):
    '''
    INPUT:
        dataset : a dataset with dimension [nDatapoints x nNodes]
    OUTPUT:
        M : matrix of partial correlation coefficients
    '''
    D = dataset
    nNodes = D.shape[1]
    #allocate memory
    M = np.zeros((nNodes,nNodes))
    #compute the partial correlation of x and each remaining variable y, conditioning on all except x and y 
    for x in range(nNodes-1):
        for y in range(x+1, nNodes):
            #create some indices
            idx = np.ones(nNodes, dtype=np.bool)
            #to not include x and y on the regressors
            idx[x] = False 
            idx[y] = False

            #regressed out the rest of the variables from x and from y, independently
            reg_x = LinearRegression().fit(D[:,idx], D[:,x])
            reg_y = LinearRegression().fit(D[:,idx], D[:,y])
            #compute the residuals for x and for y
            #residual = x - Z*B_hat, 
            #where B_hat are the estimated coefficients and Z the data for the rest of the variables
            res_x = D[:,x] - np.dot(D[:,idx],reg_x.coef_)
            res_y = D[:,y] - np.dot(D[:,idx],reg_y.coef_)
            #compute the correlation of the residuals which are equal to
            #the partial correlation of x and y conditioning on the rest of the variables
            parcorr = np.corrcoef(res_x, res_y, rowvar=False)[0,1] 
            #partial_correlation is symmetric, meaning that:
            M[x,y] = M[y,x] = parcorr
   
    return M