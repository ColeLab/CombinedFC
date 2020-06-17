#Computes the Zalpha cutoff value from a standard normal distribution N(mean=0,std.dev=1)
#that is used to reject or not the null hypothesis Ho.
#the function works for single values or matrices.

from scipy import stats

def Zcutoff(alpha = 0.01, kind = 'two-sided'):
    '''
    INPUT:
        alpha : level of significance, a value between (1 and 0). Lower alpha implies higher Zalpha. Default = 0.01.
        kind : 'two-sided', 'one-sided-right' or 'one-sided-left'. Depending on the alternative hypothesis Ha. See below.
    OUTPUT:
        Zalpha : the Z cutoff statistic for the input alpha and kind of alternative hypothesis Ha.
    '''

    if kind == 'two-sided':
        #For null Ho: r = 0 and alternative Ha: r != 0, where r  is correlation/partial correlation coefficient. 
        #using cumulative distribution function (cdf) of the standard normal
        #alpha = 2*(1-cdf(Zalpha)), solve for Zalpha = inverse_of_cdf(1-alpha/2)
        #Zalpha defines the null hypothesis rejection regions such that:
        #reject the null hypothesis Ho if Fz >= +Zalpha or Fz <= -Zalpha
        #Fz is the Fisher z-transform of the r value observed
        #(from scipy use stats.norm.ppf to compute the inverse_of_cdf)
        Zalpha = stats.norm.ppf(1-alpha/2,loc=0,scale=1)
    
    elif kind == 'one-sided-right':
        #For null Ho: r = 0 and alternative Ha: r > 0 
        #alpha = 1 - cdf(Zalpha), solve for Zalpha = inverse_of_cdf(1-alpha)
        #reject the null hypothesis if Fz >= +Zalpha
        Zalpha = stats.norm.ppf(1-alpha,loc=0,scale=1)
        
    elif kind == 'one-sided-left':
        #For null Ho: r = 0 and alternative Ha: r < 0
        #alpha = cdf(Zalpha), solve for Zalpha = inverse_of_cdf(alpha)
        #reject the null hypothesis if Fz <= -Zalpha
        Zalpha = stats.norm.ppf(alpha,loc=0,scale=1)
        
    return Zalpha