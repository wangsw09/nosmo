import numpy as np

def l2_reg_obj(x):
    return 0.5 * np.sum(x ** 2)

def l2_reg_grad(x):
    return x

def l2_reg_prox(x, tau, **kwargs):
    return x / (1.0 + tau)
