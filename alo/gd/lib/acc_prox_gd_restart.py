import warnings

import numpy as np
import numpy.linalg as npla

def acc_prox_gd_restart(x, grad_func, prox_func, ss, lam, obj_func, max_iter, tol, **kwargs):
    """
    fista
    """
    if type(x) is np.ndarray:
        return acc_prox_gd_restart_arr(x, grad_func, prox_func, ss, lam, obj_func, max_iter, tol, **kwargs)
    else:
        return acc_prox_gd_restart_1d(x, grad_func, prox_func, ss, lam, obj_func, max_iter, tol, **kwargs)

def acc_prox_gd_restart_1d(x, grad_func, prox_func, ss, lam, obj_func, max_iter, tol, **kwargs):
    """
    x0 must be either scalar or np.array
    """
    
    x0 = x + 2.0 * tol
    y = 0.0
    k = 0
    iter_count = 0
    
    while abs(x - x0) > tol:
        k += 1
        iter_count += 1
        y = x + (k - 1.0) / (k + 2.0) * (x - x0)
        x0 = x
        x = prox_func( y - ss * grad_func(y, **kwargs["grad"]), ss * lam, **kwargs["prox"] )
        if (y - x) * (x - x0) > 0:
            k = 1

        if iter_count >= max_iter:
            warnings.warn("max_iter={0} is reached; result may not be stable".format(max_iter), RuntimeWarning)
            break
    return x

def acc_prox_gd_restart_arr(x, grad_func, prox_func, ss, lam, obj_func, max_iter, tol, **kwargs):
    """
    x0 must be either scalar or np.array
    """
    x0 = x + 2.0 * tol
    y = 0.0
    k = 0
    iter_count = 0

    while npla.norm(x - x0) > tol:
        k += 1
        iter_count += 1
        y = x + (k - 1.0) / (k + 2.0) * (x - x0)
        np.copyto(x0, x)
        x = prox_func( y - ss * grad_func(y, **kwargs["grad"]), ss * lam, **kwargs["prox"] )
        if np.dot(y - x, x - x0) > 0:
            k = 1
        
        if iter_count > max_iter:
            warn_str = "max_iter={0} is reached; result may not be stable".format(max_iter)
            warnings.warn(warn_str, RuntimeWarning)
            break
    return x

