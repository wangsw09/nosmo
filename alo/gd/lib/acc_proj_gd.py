import warnings

import numpy as np
import numpy.linalg as npla

def acc_proj_gd(x, grad_func, proj_func, ss, obj_func, max_iter, tol, **kwargs):
    """
    fista
    """
    if type(x) is np.ndarray:
        return acc_proj_gd_arr(x, grad_func, proj_func, ss, obj_func, max_iter, tol, **kwargs)
    else:
        return acc_proj_gd_1d(x, grad_func, proj_func, ss, obj_func, max_iter, tol, **kwargs)

def acc_proj_gd_1d(x, grad_func, proj_func, ss, obj_func, max_iter, tol, **kwargs):
    """
    x0 must be either scalar or np.array
    """
    
    x0 = x + 2.0 * tol
    y = 0.0
    k = 0
    
    while abs(x - x0) > tol:
        k += 1
        y = x + (k - 1.0) / (k + 2.0) * (x - x0)
        x0 = x
        x = proj_func( y - ss * grad_func(y, **kwargs["grad"]), **kwargs["proj"] )

        if k >= max_iter:
            warnings.warn("max_iter={0} is reached; result may not be stable".format(max_iter), RuntimeWarning)
            break
    return x

def acc_proj_gd_arr(x, grad_func, proj_func, ss, obj_func, max_iter, tol, **kwargs):
    """
    x0 must be either scalar or np.array
    """
    x0 = x + 2.0 * tol
    y = 0.0
    k = 0

    while npla.norm(x - x0) > tol:
        k += 1
        y = x + (k - 1.0) / (k + 2.0) * (x - x0)
        np.copyto(x0, x)
        x = proj_func( y - ss * grad_func(y, **kwargs["grad"]), **kwargs["proj"] )
        
        if k > max_iter:
            warnings.warn("max_iter={0} is reached; result may not be stable".format(max_iter), RuntimeWarning)
            break
    return x

