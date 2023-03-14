import numpy as np
from numpy import linalg
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def DesignMatrix(x, y, order):
    '''
    Returning design matrix on form [1, x, y, x^2, y^2, xy, ...] up to 'order'th order
    '''
    order += 1
    Matrix = []
    
    for i in range(order):
        jmax = int(np.ceil((i+1)/2))
        
        for j in range(jmax):
            Matrix.append(x**(i-j) * y**j)
            
            if j != i-j:
                Matrix.append(x**j * y**(i-j))

    return np.array(Matrix).T



def OLS_matrix_inversion(X, data):
    '''
    Performing ordinary least squares by matrix inversion
    '''
    XTX = (X.T @ X)
    XTX += np.eye(len(XTX)) * 1e-12   # avoiding singular matrix
    beta = linalg.inv(XTX) @ X.T @ data
    
    return beta




def error_bias2_variance(X, data, degree, n_bootstraps=500, test_size=0.2):
    '''
    Performing bias variance analysis and returning error, bias^2 and variance
    '''

    # splitting data and design matrix in test and train 
    X_train, X_test, y_train, y_test = train_test_split(X, data, test_size=test_size)
        
    # empty array for computing expectation value
    y_pred = np.zeros([n_bootstraps, y_test.shape[0], y_test.shape[1]])
        
    for i in range(n_bootstraps):
        X_, y_ = resample(X_train, y_train)
        beta = OLS_matrix_inversion(X_, y_)
        y_pred[i] = X_test @ beta

    error = np.mean( np.mean((y_test - y_pred)**2, axis=0, keepdims=True) )
    bias2 = np.mean( (y_test - np.mean(y_pred, axis=0, keepdims=True))**2 )
    variance = np.mean( np.var(y_pred, axis=0, keepdims=True) )
            
    return error, bias2, variance



def bias_variance_tradeoff(data, n_bootstraps=500, max_degree=10, test_size=0.2):
    '''
    Bias-variance tradeoff while looping through model complexity
    '''
    error = np.zeros(max_degree)
    bias2 = np.zeros(max_degree)
    var = np.zeros(max_degree)
    
    arr0 = np.linspace(0, 1, data.shape[0], endpoint=False)
    arr1 = np.linspace(0, 1, data.shape[1], endpoint=False)

    for deg in range(max_degree):
        # defining design matrix
        X = DesignMatrix(arr0, arr1, deg)
        
        error[deg], bias2[deg], var[deg] = error_bias2_variance(X, data, deg, n_bootstraps=n_bootstraps, test_size=test_size)
            
    return error, bias2, var