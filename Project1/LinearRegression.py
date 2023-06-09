import numpy as np
from numpy import linalg
from sklearn.linear_model import Lasso

def OLS_matrix_inversion(X, y, lmbd=0):
    '''
    Performing ordinary least squares by matrix inversion
    X: design matrix
    y: data
    '''
    XTX = (X.T @ X)
    XTX += np.eye(len(XTX)) * 1e-12   # avoiding singular matrix
    beta = linalg.inv(XTX) @ X.T @ y

    return beta



def OLS_SVD(X, y, lmbd=0):
    '''
    Performing ordinary least squares by singular matrix decomposition. Does not work.
    X: design matrix
    y: data
    '''
    XTX = X.T @ X
    U, S, VT = linalg.svd(XTX)
    D_inv = np.diag(1/S)
    
    XTX_inv = VT.T @ D_inv @ U.T
    beta = XTX_inv @ X.T @ y

    return beta



def ridge_regression(X, y, lmbd):
    '''
    Performing Ridge regression with matrix inversion
    '''
    XTX = (X.T @ X)
    XTX += lmbd * np.eye(len(XTX))   # Ridge
    beta = linalg.inv(XTX) @ X.T @ y
    
    return beta



def lasso_regression(X_test, X_train, y_train, lmbd):

    '''
    Performing Lasso regression with matrix inversion

    '''
    LassoReg = Lasso(lmbd)
    LassoReg.fit(X_train, y_train)
    y_pred = LassoReg.predict(X_test)
    
    return y_pred



def compute_prediction(X_test, X_train, y_train, params, method):
    """
    Computing predictions using the provided regression method.
    """
    if method is lasso_regression:
        return method(X_test, X_train, y_train, params)

    else:
        beta = method(X_train, y_train, params)

    return np.dot(X_test, beta)
