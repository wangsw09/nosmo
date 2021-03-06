import warnings

import numpy as np
import numpy.linalg as npla

def acc_proj_gd_restart(x, grad_func, proj_func, ss, obj_func, max_iter, tol, **kwargs):
    """
    fista
    """
    if type(x) is np.ndarray:
        return acc_proj_gd_restart_arr(x, grad_func, proj_func, ss, obj_func, max_iter, tol, **kwargs)
    else:
        return acc_proj_gd_restart_1d(x, grad_func, proj_func, ss, obj_func, max_iter, tol, **kwargs)

def acc_proj_gd_restart_1d(x, grad_func, proj_func, ss, obj_func, max_iter, tol, **kwargs):
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
        x = proj_func( y - ss * grad_func(y, **kwargs["grad"]), **kwargs["proj"] )
        if (y - x) * (x - x0) > 0:
            k = 1

        if iter_count >= max_iter:
            warnings.warn("max_iter={0} is reached; result may not be stable".format(max_iter), RuntimeWarning)
            break
    return x

def acc_proj_gd_restart_arr(x, grad_func, proj_func, ss, obj_func, max_iter, tol, **kwargs):
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
        x = proj_func( y - ss * grad_func(y, **kwargs["grad"]), **kwargs["proj"] )
        if np.dot(y - x, x - x0):
            k = 1
        
        if iter_count > max_iter:
            warnings.warn("max_iter={0} is reached; result may not be stable".format(max_iter), RuntimeWarning)
            break
    return x

