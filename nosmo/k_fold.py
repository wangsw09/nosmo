import numpy as np
import numpy.random as npr

def k_fold(y, X, k, lam, func, **kwargs):
    n, p = X.shape
    idx = np.arange(n)
    npr.shuffle(idx)

    mask = np.ones(n, dtype=np.bool)
    sep = np.linspace(0, n, k + 1).astype(np.int32)
    err = 0.0
    for i in range(k):
        mask[sep[i] : sep[i + 1]] = False
        beta = func(y=y[mask], X=X[mask, :], lam=lam, **kwargs)
        pred = np.dot(X[sep[i] : sep[i+1], :], beta)
        err += np.sum((y[sep[i] : sep[i+1]] - pred) ** 2)
        mask[sep[i] : sep[i + 1]] = True
    err /= n
    return err

def k_fold_vec(y, X, k, lams, func, **kwargs):
    n, p = X.shape
    idx = np.arange(n)
    npr.shuffle(idx)

    mask = np.ones(n, dtype=np.bool)
    sep = np.linspace(0, n, k + 1).astype(np.int32)
    err = np.zeros(lams.shape[0])
    for i in range(k):
        mask[sep[i] : sep[i + 1]] = False
        beta = func(y=y[mask], X=X[mask, :], lams=lams, **kwargs)
        pred = np.dot(X[sep[i] : sep[i+1], :], beta)
        err += np.sum((y[sep[i] : sep[i+1], np.newaxis] - pred) ** 2, axis=0)
        mask[sep[i] : sep[i + 1]] = True
    err /= n
    return err

