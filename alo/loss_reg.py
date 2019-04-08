import numpy as np

def glm_loss(y, u, family="gaussian"):
    n = y.shape[0]
    if family == "gaussian":
        return (u - y), np.ones(n)
    elif family == "binomial":
        tmp = np.exp( - u)
        return -y + 1.0 / (1.0 + tmp), tmp / ((1.0 + tmp) ** 2)
    elif family == "poisson":
        tmp = np.exp(u)
        return -y + tmp, tmp
    else:
        raise ValueError("family type undefined.")
