import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
import sklearn as skl
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from mpl_toolkits.mplot3d import Axes3D


def f(x,y):

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        
    return  term1 + term2 + term3 + term4


"""
This section of code generates training data for fitting the Gaussian Process Regression model

Variables declared:
x_end: The length till which x's need to be sampled
num_train: It denotes the number of training points that needs to be extracted within the define interval
sigma_noise: It is the standard deviation of the normal distribution from which random numbers are sampled

"""


# Defining train data set
step_size = 0.05
arr = np.arange(0, 10, step_size)
Mat = np.meshgrid(arr, arr)
f_x = f(*Mat)

# Adding noise to the functional evaluations
num_train = 200
sigma_noise = 0.4
error_train = np.random.normal(loc=0, scale=sigma_noise, size=num_train)
y_train = f_x + error_train

#test data

step_size = 0.05
arr = np.arange(10, 20, step_size)
Mat = np.meshgrid(arr, arr)
x_test = f(*Mat)

##########################################################################################

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#compute z
z = FrankeFunction(x, y)
# create labels and features 
features = np.stack([np.ndarray.flatten(x), np.ndarray.flatten(y)], axis=1)
labels = np.ndarray.flatten(z)

# Plot Franke Function
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

"""
The part of code is used to describe the type of kernel used and the gaussian process regression model

1. Definition of kernel:

kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds=(1e-2, 1e2)) \
              * RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2))

a. RBF kenel has been used for this problem.
b. Hyperparameter definitions:
    (i) sigma_f: defines the amplitude of the kernel function
    (ii) l: the locality parameter; used to define how far each point is able to interacts
c. The hyperparameter are best chosen so as to minimise the marginal log likelihood

2. Definition of gp model

gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n**2, n_restarts_optimizer=10, )

a. Parameter definitions
    (i) sigma_n: It is the value added to the diagonal elements of defined  kernel matrix. Larger values mean a 
        larger value of noise in the data
    (ii) The number of restarts of the optimizer for finding the kernel’s parameters which maximize the 
        log-marginal likelihood

3. Output of the code is the gp model and the predictions on the test data

"""

# Initial values of l, sigma_f and sigma_n needs to be defined.
# Other inputs are the training and test datasets that need to be input

def gpPrediction( l, sigma_f, sigma_n , X_train, y_train, X_test):
  # Kernel definition 
  kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds=(1e-2, 1e2)) \
              * RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2))
  # GP model 
  gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n**2, n_restarts_optimizer=10, )
  # Fitting in the gp model
  gp.fit(X_train, y_train)
  # Make the prediction on test set.
  y_pred = gp.predict(X_test)
  return y_pred, gp

"""
l_init and sigma_f init are the initial values of the hyperparameters l and sigma_f of the kernel function

"""

l_init = 1
sigma_f_init = 3
sigma_n = 1

y_pred, gp = gpPrediction(l_init, sigma_f_init, sigma_n , X_train, y_train, X_test)



# Generate samples from posterior distribution. 
y_hat_samples = gp.sample_y(X_test, n_samples=200)
# Compute the mean of the sample. 
y_hat = np.apply_over_axes(func=np.mean, a=y_hat_samples, axes=1).squeeze()
# Compute the standard deviation of the sample. 
y_hat_sd = np.apply_over_axes(func=np.std, a=y_hat_samples, axes=1).squeeze()

"""
This portion of the code is used to visualize the preductions made by gp model on the test datase. 
Cridible intervals for the predictions has also been plotted and indicated through the tranparent green corridor
"""

""" fig, ax = plt.subplots(figsize=(15, 8))
# Plotting the training data.
sns.scatterplot(x=X_train, y=y_train, label='training data', ax=ax)
# Plot the functional evaluation
sns.lineplot(x=X_test, y=f(X_test), color='red', label='f(x)', ax=ax)
# Plot corridor. 
ax.fill_between(x=X_test, y1=(y_hat - 2*y_hat_sd), y2=(y_hat + 2*y_hat_sd), color='green',alpha=0.3, label='Credible Interval')
# Plot prediction. 
sns.lineplot(x=X_test, y=y_pred, color='green', label='pred')

# Labeling axes
ax.set(title='Gaussian Process Regression')
ax.legend(loc='lower left')
ax.set(xlabel='x', ylabel='')
plt.show() """


""" fig = plt.figure(figsize=(7, 5))
ax = Axes3D(fig)
ax.plot_surface(test_d1, test_d2, z, cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
ax.scatter(np.asarray(train_X)[:,0], np.asarray(train_X)[:,1], train_y, c=train_y, cmap=cm.coolwarm)
ax.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.6)
ax.set_title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"])) """


surf = ax.plot_surface(X_test, y_test, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()