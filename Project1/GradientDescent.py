import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso


import autograd.numpy as np
from autograd import grad

from Statistics import norm2
from LinearRegression import OLS_matrix_inversion



def GradientDescent(X, y, learning_rate, momentum=0, eps=1e-8, max_iter=100_000):
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
    grad = (2.0 / ndata) * X.T @ (X @ beta - y)

    # count number of iteration
    n_iter = 0
    while norm2(grad) > eps:
        grad_prev = grad
        grad = (2.0 / ndata) * X.T @ (X @ beta - y)
        beta = beta - momentum * grad_prev - learning_rate * grad

        # break if takes too long
        if n_iter > max_iter:
            break

        n_iter += 1

    # compute cost
    cost = norm2(X @ beta - y) / ndata

    return beta, n_iter, cost



def StochasticGradientDescent(X, data, momentum=0, nepochs=10, batch_size=5, eps=1e-8):
    '''
    Gradient descent with fixed learning rate
    X: column vector
    data:
    gamma: learning rate

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

    # compute cost
    cost = norm2(X @ beta - data) / ndata

    return beta, cost



def OLSGradientDescent(X, y, learning_rate, momentum=0, eps=1e-8, max_iter=100_000, adaptive_method="none"):
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
    def fCost(beta):
        return np.sum( (y - np.dot(X, beta))**2 ) / len(y)

    ndata, ndims = np.shape(X)
    rho = 0.9

    # initial guesses for beta and the gradient
    if adaptive_method == "none":
        s = 1-eps
    else:
        s = 0
        
    beta = np.random.rand(ndims)
    gradient = grad(fCost)

    # count number of iteration
    n_iter = 0
    while norm2(gradient(beta)) > eps:
        beta_prev = beta
        s_prev = s
        grad_beta = gradient(beta)

        if adaptive_method == "adagrad":
            s = s_prev + grad_beta**2

        elif adaptive_method == "rmsprop":
            s = s_prev * rho + (1 - rho) * grad_beta**2
        beta = beta - momentum * gradient(beta_prev) - learning_rate * grad_beta / (np.sqrt(s) + eps)

        # break if takes too long
        if n_iter > max_iter:
            break

        n_iter += 1

    # compute cost
    cost = norm2(X @ beta - y) / ndata

    return beta, n_iter, cost



def RidgeGradientDescent(X, y, lmbd, learning_rate, momentum=0, eps=1e-8, max_iter=100_000, adaptive_method="none",):
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
    def CostRidge(beta):
        return np.sum( (y - np.dot(X, beta))**2 ) / len(y) + lmbd * np.sum( beta**2 )

    ndata, ndims = np.shape(X)
    rho = 0.9

    # initial guesses for beta and the gradient
    if adaptive_method == "none":
        s = 1-eps
    else:
        s = 0
        
    beta = np.random.rand(ndims)
    gradient = grad(CostRidge)

    # count number of iteration
    n_iter = 0
    while norm2(gradient(beta)) > eps:
        beta_prev = beta
        s_prev = s
        grad_beta = gradient(beta)

        if adaptive_method == "adagrad":
            s = s_prev + grad_beta**2

        elif adaptive_method == "rmsprop":
            s = s_prev * rho + (1 - rho) * grad_beta**2
        beta = beta - momentum * gradient(beta_prev) - learning_rate * grad_beta / (np.sqrt(s) + eps)

        # break if takes too long
        if n_iter > max_iter:
            break

        n_iter += 1

    # compute cost
    cost = norm2(X @ beta - y) / ndata

    return beta, n_iter, cost



def OLSStochasticGradientDescent(X, y, learning_rate, momentum=0, nepochs=10, batch_size=5, eps=1e-8, adaptive_method="none"):
    '''
    Gradient descent with fixed learning rate
    X: column vector
    data:
    gamma: learning rate

    Tunable learning rate
    '''
    def fCost(beta):
        return np.sum( (y - np.dot(X, beta))**2 ) / len(y)
    
    # ensure that batch_size is compatible with the size of our data
    ndata, ndims = np.shape(X)
    assert(ndata % batch_size == 0)
    nbatches = int(ndata / batch_size)
    rho = 0.9

    # splitting up data in minibatches
    X_batch = X.reshape(nbatches, batch_size, ndims)
    data_batch = y.reshape(nbatches, batch_size)

    # initial guesses for beta and the gradient
    if adaptive_method == "none":
        s = 1-eps
    else:
        s = 0
        
    beta = np.random.rand(ndims)
    gradient = grad(fCost)

    for epoch in range(nepochs):
        for batch in range(nbatches):
            # pick random minibatch
            idx = np.random.randint(nbatches)
            X_, y_ = X_batch[idx], data_batch[idx]

            def fCost(beta):
                return np.sum( (y_ - np.dot(X_, beta))**2 ) / len(y_)
        
            # update beta
            beta_prev = beta
            s_prev = s
            grad_beta = gradient(beta)
            
            if adaptive_method == "adagrad":
                s = s_prev + grad_beta**2

            elif adaptive_method == "rmsprop":
                s = s_prev * rho + (1 - rho) * grad_beta**2
            beta = beta - momentum * gradient(beta_prev) - learning_rate * grad_beta / (np.sqrt(s) + eps)

    # compute cost
    cost = norm2(X @ beta - y) / ndata

    return beta, cost



def RidgeStochasticGradientDescent(X, y, learning_rate, lmbd, momentum=0, nepochs=10, batch_size=5, eps=1e-8, adaptive_method="none"):
    '''
    Gradient descent with fixed learning rate
    X: column vector
    data:
    gamma: learning rate

    Tunable learning rate
    '''
    def fCost(beta):
        return np.sum( (y - np.dot(X, beta))**2 ) / len(y) + lmbd * np.sum( beta**2 )

    
    # ensure that batch_size is compatible with the size of our data
    ndata, ndims = np.shape(X)
    assert(ndata % batch_size == 0)
    nbatches = int(ndata / batch_size)
    rho = 0.9

    # splitting up data in minibatches
    X_batch = X.reshape(nbatches, batch_size, ndims)
    data_batch = y.reshape(nbatches, batch_size)

    # initial guesses for beta and the gradient
    if adaptive_method == "none":
        s = 1-eps
    else:
        s = 0
        
    beta = np.random.rand(ndims)
    gradient = grad(fCost)


    for epoch in range(nepochs):
        for batch in range(nbatches):
            # pick random minibatch
            idx = np.random.randint(nbatches)
            X_, y_ = X_batch[idx], data_batch[idx]

            def fCost(beta):
                return np.sum( (y - np.dot(X_, beta))**2 ) / len(y_) + lmbd * np.sum( beta**2 )
        
            # update beta
            beta_prev = beta
            s_prev = s
            grad_beta = gradient(beta)

            if adaptive_method == "adagrad":
                s = s_prev + grad_beta**2

            elif adaptive_method == "rmsprop":
                s = s_prev * rho + (1 - rho) * grad_beta**2
            
            beta = beta - momentum * gradient(beta_prev) - learning_rate * grad_beta / (np.sqrt(s) + eps)


    # compute cost
    cost = norm2(X @ beta - y) / ndata

    return beta, cost


# for rmsprop: rms=0.9, learning_rate = 1e-3