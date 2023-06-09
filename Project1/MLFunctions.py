import numpy as np
from numpy import linalg
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.linear_model import Lasso

from LinearRegression import OLS_matrix_inversion, compute_prediction




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



def bootstrap(X, data, n_bootstraps=500, test_size=0.2, regression_method=OLS_matrix_inversion, regression_params=0):
    '''
    Performing bootstrapping, returning error, bias^2 and variance
    '''
    # splitting data and design matrix in test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, data, test_size=test_size)
        
    # empty array for computing expectation value
    y_pred = np.zeros((n_bootstraps, y_test.shape[0], y_test.shape[1]))
        
    for i in range(n_bootstraps):
        X_, y_ = resample(X_train, y_train)
        y_pred[i] = compute_prediction(X_test, X_, y_, regression_params, regression_method)

    # compute expectation value of model prediction
    Exp_y = np.mean(y_pred, axis=0) #, keepdims=True)

    # compute error, bias and variance
    error =    np.mean( (y_test - y_pred)**2 )
    bias2 =    np.mean( (y_test -  Exp_y)**2 )
    variance = np.mean( (y_pred -  Exp_y)**2 )
    
    return error, bias2, variance



def bias_variance_tradeoff(data, design_arr, n_bootstraps=500, max_degree=10, test_size=0.2, regression_method=OLS_matrix_inversion, regression_params=0):
    '''
    Bias-variance tradeoff while looping through model complexity
    '''
    error = np.zeros(max_degree)
    bias2 = np.zeros(max_degree)
    var = np.zeros(max_degree)

    for deg in range(max_degree):
        # defining design matrix
        X = DesignMatrix(design_arr, design_arr, deg)
        
        error[deg], bias2[deg], var[deg] = bootstrap(X, data, n_bootstraps=n_bootstraps, test_size=test_size)
        
    return error, bias2, var



def cross_validation(X, data, test_size=0.2, regression_method=OLS_matrix_inversion, regression_params=0):
    '''
    Performing cross validation, returning mse
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

        y_pred.append(compute_prediction(X_test, X_train, y_train, regression_params, regression_method))
        y_test.append(data[test_index])

    # compute expectation value of model prediction
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    # compute error, bias and variance
    error =  np.mean( (y_test - y_pred)**2 )
            
    return error


def GradientDescent(X, data, learning_rate, momentum=0, eps=1e-8, max_iter=100_000):
    '''
    Gradient descent with fixed learning rate

    Inputs
    X: column vector
    data:
    learning_rate: learning rate
    momentum: rate of memory

    Outputs
    beta:
    niter:
    cost:
    '''
    ndata, ndims = np.shape(X)

    # initial guesses for beta and the gradient
    beta = np.random.rand(ndims)
    grad = (2.0 / ndata) * X.T @ (X @ beta - data)

    # count number of iteration
    n_iter = 0
    while norm2(grad) > eps:
        grad_prev = grad
        grad = (2.0 / ndata) * X.T @ (X @ beta - data)
        beta = beta - momentum * grad_prev - learning_rate * grad

        # break if takes too long
        if n_iter > max_iter:
            break

        n_iter += 1

    # compute cost
    cost = norm2(X @ beta - data) / ndata

    return beta, n_iter, cost



def StochasticGradientDescent(X, data, momentum=0, nepochs=10, batch_size=5, eps=1e-8, cost='norm2'):
    '''
    Gradient descent with fixed learning rate
    X: column vector
    data:
    gamma: learning rate
    cost: if 'sigmoid' it's sigmoid, else norm2
    Tunable learning rate
    '''
    # ensure that batch_size is compatible with the size of our data
    ndata, ndims = np.shape(X)
    assert(ndata % batch_size == 0)
    nbatches = int(ndata / batch_size)

    # splitting up data in minibatches
    X_batch = X.reshape(nbatches, batch_size, ndims)
    data_batch = data.reshape(nbatches, batch_size)

    # initial guesses for beta and the gradient
    beta = np.random.rand(ndims)
    grad = (2.0 / ndata) * X.T @ (X @ beta - data)

    # compute learning rate
    H = (2 / ndata) *  X.T @ X
    eigVal, _ = np.linalg.eig(H)
    learning_rate = 1 / np.max(eigVal)


    for epoch in range(nepochs):
        for batch in range(nbatches):
            # pick random minibatch
            idx = np.random.randint(nbatches)
            X_, data_ = X_batch[idx], data_batch[idx]

            # computing gradient over minibatch
            grad_prev = grad
            grad = (2.0 / batch_size) * X_.T @ (X_ @ beta - data_)
        
            # update beta
            beta = beta - momentum * grad_prev - learning_rate * grad
            # print(beta)

    # compute cost
    if cost == 'sigmoid':
        cost = sigmoid(X)
    else:
        cost = norm2(X @ beta - data) / ndata

    return beta, cost

