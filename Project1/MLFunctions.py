import numpy as np
from numpy import linalg
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.linear_model import Lasso

from Statistics import MSE

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



def OLS_SVD(X, data):
    '''
    Performing ordinary least squares by singular matrix decomposition
    '''
    XTX = X.T @ X
    U, S, VT = linalg.svd(XTX)
    D_inv = np.diag(1/S)
    
    XTX_inv = VT.T @ D_inv @ U.T
    beta = XTX_inv @ X.T @ data

    return beta



def ridge_regression(X, data, lmbd):
    '''
    Performing Ridge regression with matrix inversion
    '''
    XTX = (X.T @ X)
    XTX += lmbd * np.eye(len(XTX))   # Ridge
    beta = linalg.inv(XTX) @ X.T @ data
    
    return beta



def lasso_regression(X, data, lmbd):
    '''
    Performing Lasso regression with matrix inversion

    '''
    LassoReg = Lasso(lmbd)
    LassoReg.fit(X,data)
    y_pred = LassoReg.predict(X)
    
    beta = LassoReg.coef_
    
    return beta, y_pred



def bootstrap(X, data, n_bootstraps=500, test_size=0.2):
    '''
    Performing bootstrapping, returning error, bias^2 and variance
    '''

    # splitting data and design matrix in test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, data, test_size=test_size)
        
    # empty array for computing expectation value
    y_pred = np.zeros([n_bootstraps, y_test.shape[0], y_test.shape[1]])
        
    for i in range(n_bootstraps):
        X_, y_ = resample(X_train, y_train)
        beta = OLS_matrix_inversion(X_, y_)
        y_pred[i] = X_test @ beta

    # compute expectation value of model prediction
    Exp_y = np.mean(y_pred, axis=0) #, keepdims=True)

    # compute error, bias and variance
    error =    np.mean( (y_test - y_pred)**2 )
    bias2 =    np.mean( (y_test -  Exp_y)**2 )
    variance = np.mean( (y_pred -  Exp_y)**2 )
            
    return error, bias2, variance



def cross_validation(X, data, test_size=0.2):
    '''
    Performing cross validation, returning error, bias^2 and variance
    '''

    # splitting data and design matrix in test and train sets
    n_splits = int(1 / test_size)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)  

    #y_pred = np.zeros([n_splits, y_test.shape[0], y_test.shape[1]])
    y_test = []
    y_pred = []
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], data[train_index]
        X_test = X[test_index]

        beta = OLS_matrix_inversion(X_train, y_train)

        y_test.append(data[test_index])
        y_pred.append(X_test @ beta)

    # compute expectation value of model prediction
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    Exp_y = np.mean(y_pred, axis=0) #, keepdims=True)

    # compute error, bias and variance
    error =    np.mean( (y_test - y_pred)**2 )
    bias2 =    np.mean( (y_test - Exp_y)**2 )
    variance = np.mean( (y_pred - Exp_y)**2 )
            
    return error, bias2, variance



def bias_variance_tradeoff(data, design_arr, resampling_method, method_params=0, max_degree=10, test_size=0.2):
    '''
    Bias-variance tradeoff while looping through model complexity
    '''
    error = np.zeros(max_degree)
    bias2 = np.zeros(max_degree)
    var = np.zeros(max_degree)

    for deg in range(max_degree):
        # defining design matrix
        X = DesignMatrix(design_arr, design_arr, deg)
        
        if resampling_method == 'bootstrap':
            error[deg], bias2[deg], var[deg] = bootstrap(X, data, n_bootstraps=method_params, test_size=test_size)
        
        elif resampling_method == "cross-validation":
            error[deg], bias2[deg], var[deg] = cross_validation(X, data, test_size=test_size)
            
    return error, bias2, var
