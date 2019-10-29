#compute the linear simple regression Y = a + bX + E
#and a significance test for the b coefficient

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

def simpleRegressionSig(y, x, alpha = 0.01, sigTest = True):
    '''
    INPUT:
        y : vector [nDatapoints x 1] with node y data, for the regression y = a + bx + E
        x : vector [nDatapoints x 1] with regressor node x data
        alpha : cutoff for the significance decision. Default is 0.01
        sigTest : if True, perform the t-test of significance for the Beta coefficients
    OUTPUT:
        b : the b regression coefficient. If sigTest = True, return b if significant, otherwise return 0
            
    '''
    
    #check dimensions, if < 2, expand to get [nDatapoints x 1]
    if len(y.shape) < 2:
        y = np.expand_dims(y, axis=1)
    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
    
    nDatapoints = y.shape[0]

    reg_y = LinearRegression().fit(x, y)
        
    if sigTest == True:
        #parameters estimated =  intercept and the beta coefficients
        params = np.append(reg_y.intercept_,reg_y.coef_)
        #number of parameters estimated
        nParams = len(params)
        #obtain predicted data y
        y_hat = reg_y.predict(x)
        #append a column of 1s (for intercept) to the regressor data
        newR = np.append(np.ones((nDatapoints,1)), x, axis=1)
        #mean squared errors MSE, adjusted by 1/(datapoints - num.parameters)
        MSE = (np.sum(np.square(y - y_hat)))/(nDatapoints - nParams)
        #compute variance of parameters (intercept and betas)
        var_params = MSE*(np.linalg.inv(np.dot(newR.T,newR)).diagonal())
        #compute standard deviation
        std_params = np.sqrt(var_params)
        #transform parameters into t-statistics
        ts_params = params/std_params
        #p-value for a t-statistic in a two-sided one sample t-test
        p_values = [2*(1-stats.t.cdf(np.abs(i),df = nDatapoints-1)) for i in ts_params]

        #remove the intercept p-value
        p_value = np.delete(p_values,0)
        #record the Betas with p-values < alpha
        b = np.multiply(reg_y.coef_, p_value < alpha)

    if sigTest == False:
        #save the beta coefficients in the corresponding x row without significance test
        b = reg_y.coef_

    return b