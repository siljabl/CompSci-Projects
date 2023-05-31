import numpy as np

def MSE(data, model):
    'Mean square error'
        
    return np.mean( (data - model)**2 )


def R2(data, model):
    'R² score function'
    nom = ( (data - model)**2 ).sum()
    denom = ( (data - data.mean())**2 ).sum()
    
    return 1 - nom / denom



def norm2(M):
    return np.sqrt(np.sum(M**2))



def sigmoid(x):
    return 1 / (1 + np.exp(-x))