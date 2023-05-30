import ultranest
from ultranest.plot import cornerplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

#import sample data
data = pd.read_csv('C:/Users/franscho/Documents/CompSci-Projets/Project3/Bayes/example_data.txt', header = 0)

x = data.iloc[:,0]
y = data.iloc[:,1]
y_err = data.iloc[:,2]

#plot sample data
plt.errorbar(x, y, yerr=y_err, ls=' ', marker='x')
plt.plot(x, y, ls=':', alpha=0.5, color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


param_names = ['peak_location', 'peak_height', 'peak_number']

def my_prior_transform(cube):
    params = cube.copy()

    # location: uniform prior
    lo = 400
    hi = 800
    params[0] = cube[0] * (hi - lo) + lo

    # height: log-uniform prior
    lo = 0.1
    hi = 100
    params[1] = 10**(cube[1] * (np.log10(hi) - np.log10(lo)) + np.log10(lo))

    # number: uniform  prior
    lo = 0
    hi = 50
    params[2] = cube[2] * (hi - lo) + lo

    return params

def my_likelihood(params):
    location, height, number = params

    # TODO - implement likelihood function
    
    
    # compute intensity at every x position according to the model
    y_model = height * np.exp(-0.5 * ((x - location)/number)**2)
    # compare model and data with gaussian likelihood:
    like = -0.5 * (((y_model - y)/y_err)**2).sum()
    return like


sampler = ultranest.ReactiveNestedSampler(param_names, my_likelihood, my_prior_transform)

result = sampler.run()
sampler.print_results()
sampler.plot_run()
sampler.plot_trace()
sampler.plot_corner()

cornerplot(result)