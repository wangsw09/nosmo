import numpy as np
import numpy.random as npr

def out_sample_err(y, X, X_new, y_new, lam, func, **kwargs):
    n, p = X.shape

    beta = func(y=y, X=X, lam=lam, **kwargs)
    pred = np.dot(X_new, beta)
    err = np.mean((y_new - pred) ** 2)
    return err

