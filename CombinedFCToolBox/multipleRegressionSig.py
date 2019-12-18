#compute the multiple regression for each node Xi on the rest of the variable set V\{Xi}, V={X1,...,Xp}
#e.g. X1 = b0 +b2X2 + b3X3 + ... + bpXp + E1
#build a connectivity matrix M where each row contains the Betas for the regression of Xi on the rest of the variables
#significant regression betas represent edges in the connectivity model.
#non-significant betas are zeroed out

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

def multipleRegressionSig(dataset, alpha = 0.01, sigTest = False):
    '''
    INPUT:
        dataset : in dimension [nDatapoints x nNodes]
        alpha : cutoff for the significance decision. Default is 0.01
        sigTest : if True, perform the t-test of significance for the Beta coefficients
    OUTPUT:
        M : a connectivity network of beta coefficients. If sigTest = True, only significant betas 
            Note: It is not a symmetric matrix.
    '''


    D = dataset
    nNodes = D.shape[1] #number of variables
    nDatapoints = D.shape[0]
    
    M = np.zeros((nNodes,nNodes))
    for x in range(nNodes):
        #create some indices
        idx = np.ones(nNodes, dtype=np.bool)
        #to not include x on the set of regressors, ie. V\{x}
        idx[x] = False 
        #regressed x on the rest of the variables in the set
        reg_x = LinearRegression().fit(D[:,idx], D[:,x])
        
        if sigTest == True:
            #parameters estimated =  intercept and the beta coefficients
            params = np.append(reg_x.intercept_,reg_x.coef_)
            #number of parameters estimated
            nParams = len(params)
            #obtain predicted data x
            x_hat = reg_x.predict(D[:,idx])
            #append a column of 1s (for intercept) to the regressors dataset
            newR = np.append(np.ones((nDatapoints,1)),D[:,idx],axis=1)
            #see chapter 12 and 13 of Devore's Probability textbook
            #mean squared errors MSE = SSE/(n-k-1), where k is the number of covariates
            #pg.519 Devore's
            MSE = (np.sum(np.square(D[:,x] - x_hat)))/(nDatapoints - nParams)
            #compute variance of parameters (intercept and betas)
            var_params = MSE*(np.linalg.inv(np.dot(newR.T,newR)).diagonal())
            #compute standard deviation
            std_params = np.sqrt(var_params)
            #transform parameters into t-statistics under the null of B = 0
            Bho = 0 #beta under the null
            ts_params = (params - Bho)/std_params
            #p-value for a t-statistic in a two-sided one sample t-test
            p_values = [2*(1-stats.t.cdf(np.abs(i),df = nDatapoints-1)) for i in ts_params]
            
            #remove the intercept p-value
            p_values = np.delete(p_values,0)
            #record the Betas with p-values < alpha
            M[x,idx] = np.multiply(reg_x.coef_, p_values < alpha)
        
        if sigTest == False:
            #save the beta coefficients in the corresponding x row without significance test
            M[x,idx] = reg_x.coef_

    return M
