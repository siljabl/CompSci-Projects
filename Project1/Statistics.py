import numpy as np

def MSE(data, model):
    'Mean square error'
    n = data.size
    
    return np.mean( (data - model)**2 )


def R2(data, model):
    'R² score function'
    nom = ( (data - model)**2 ).sum()
    denom = ( (data - data.mean())**2 ).sum()
    
    return 1 - nom / denom

