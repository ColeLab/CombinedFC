#Obtain the p-value for the Fisher z-transform Fz under the null hypothesis
#Fz is approximately distributed as a standard normal N(mean=0,std.dev=1)
#pvalues can be used to reject or not the null hypothesis at a chosen alpha cutoff value.
#the decision rule is: reject the null hypothesis Ho if pvalue < alpha
#the function works for single values or matrices    

from scipy import stats, linalg
import numpy as np

def pvalue(Fz, kind = 'two-sided'):
    '''
    INPUT:
        Fz : Fisher z-transformation of the correlation/partial correlation coefficient r. Use fisherZTrans() function. 
        kind : 'two-sided', 'one-sided-right' or 'one-sided-left'. Depending on the alternative hypothesis Ha. See below.
    OUTPUT:
        pval : a single p value or a matrix of p values.
    '''
  
    #Compute the p-value using the cumulative distribution function cdf of a standard normal N(mean=0,std.dev=1)
    if kind == 'two-sided':
        pval = 2*(1 - stats.norm.cdf(np.abs(Fz), loc=0, scale=1))
    elif kind == 'one-sided-right':
        pval = 1 - stats.norm.cdf(Fz, loc=0, scale=1)
    elif kind == 'one-sided-left':
        pval = stats.norm.cdf(Fz, loc=0, scale=1)
    
    return pval