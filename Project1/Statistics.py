import numpy as np

def MSE(data, model):
    n = data.size
    
    return np.mean( (data - model)**2 ) #.sum() / n


def R2(data, model):
    nom = ( (data - model)**2 ).sum()
    denom = ( (data - data.mean())**2 ).sum()
    
    return 1 - nom / denom