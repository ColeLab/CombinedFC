#implement graphical lasso with a kfold cross-validation scheme for the alpha parameter
#Glasso estimates a regularized inverse covariance, ie. precision matrix
#the entries of the precision matrix are then transformed to regularized partial correlations
#Glasso, graphical lasso, Friedman 2008 Biostatistics

from sklearn import covariance
from scipy import stats, linalg
import numpy as np
import CombinedFCToolBox as cfc

def parCorrGlasso(dataset, kfolds=10):
    '''
    INPUT:
        dataset : a dataset with dimension [nDatapoints x nNodes]
        kfolds : number of folds for the cross-validation to select regularization parameter alpha
    OUTPUT:
        Mreg : matrix of regularized (shrinked) partial correlation coefficients
        Mmod : matrix of partial correlations constrained to the glasso model
        condSetSizeMat : a matrix with the conditional sets sizes for the Mmod partial correlations
                        note: this is needed to compute the fisher z-transform for the significance tests
    '''
    D = dataset
    #in regularized methods it is necessary to standardized the data
    D = stats.zscore(D,axis=0)
    #estimate the empirical covariance
    emp_cov = np.cov(D,rowvar=False)
    #define the glasso model with cross-validation
    glasso = covariance.GraphicalLassoCV(cv = kfolds)
    #fit the model to the data and cross-validate to get the
    #best regularization parameter
    glasso.fit(D)
    #get the regularized precision matrix (inverse covariance)
    prec_mat = glasso.precision_
    #transform into regularized partial correlation coefficients
    ##https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion
    denom = np.atleast_2d(1. / np.sqrt(np.diag(prec_mat)))
    Mreg = -prec_mat * denom * denom.T
    #make the diagonal zero.
    np.fill_diagonal(Mreg,0)
    
    
    #use the connectivity model defined by glasso to compute the partial correlations between two connected nodes.
    #this is done to get non-shrinked partial correlations.
    #For adjacent nodes x and y, compute partial correlation of x and y conditional on Z = {adj(x) & adj(y)}
    #where adj(x) is the set of adjacent nodes to x in the glasso model Gm.
    
    #glasso model = non-zero entries in Mreg, ie. edges in the connectivity model
    Gm = Mreg != 0
    nNodes = Gm.shape[0]
    #allocate memory
    Mmod = np.zeros((nNodes,nNodes))
    condSetSizeMat = np.zeros((nNodes,nNodes))
    #iterate through each pair of nodes
    for x in range(nNodes-1):
        for y in range(x+1,nNodes):
            #if x and y are adjacent
            if Gm[x,y] != 0:
                #get adjacencies indices
                adj_x = np.argwhere(Gm[x,:] != 0)
                adj_y = np.argwhere(Gm[y,:] != 0)
                #get the union of adj(x) and adj(y)
                Z = np.union1d(adj_x,adj_y)
                #remove x and y from Z
                Z = Z[Z!=x]
                Z = Z[Z!=y]
                #and put them back at the beginning of Z
                Z = np.insert(Z,0,y)
                Z = np.insert(Z,0,x)
                #define a new dataset only including nodes x & y & adj(x) & adj(y)
                newD = D[:,Z]
                #compute the partial correlation for the new dataset but only get Mxy element
                #x and y will always be in the 0 and 1 positions.
                pc_xyz = cfc.parCorrInvCov(newD)[0,1]
                #is a symmetric matrix
                Mmod[x,y] = Mmod[y,x] = pc_xyz
                #get the size of the conditioning set for x and y
                condSetSizeMat[x,y] = condSetSizeMat[y,x] = len(Z)-2
                
    
    
    return Mreg, Mmod,condSetSizeMat