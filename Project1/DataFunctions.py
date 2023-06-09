import numpy as np

def FrankeFunction(x,y, noise=0):
    '''
    Franke function with and without noise
    '''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    # stochastic noise
    dim0, dim1 = np.shape(x)
    noise *= np.random.normal(0,1, [dim0, dim1])
    
    return term1 + term2 + term3 + term4 + noise



def SimpleFunction(x, a):
    '''
    Second order polynomial
    '''
    a0, a1, a2 = a
    
    return a0 + a1*x + a2*x**2