import warnings

import numpy as np
import numpy.linalg as npla

def plain_gd(x, grad_func, ss, obj_func, max_iter, tol, **kwargs):
    """
    x must be either scalar or np.array
    """
    if type(x) is np.ndarray:
        return plain_gd_arr(x, grad_func, ss, obj_func, max_iter, tol, **kwargs)
    else:
        return plain_gd_1d(x, grad_func, ss, obj_func, max_iter, tol, **kwargs)

def plain_gd_1d(x, grad_func, ss, obj_func, max_iter, tol, **kwargs):
    """
    x0 must be either scalar or np.array
    """
    
    x0 = x + 2.0 * tol
    iter_count = 0
    
    while abs(x - x0) > tol:
        x0 = x
        x = x0 - ss * grad_func(x0, **kwargs)
        iter_count += 1

        if iter_count > max_iter:
            warnings.warn("max_iter={0} is reached; result may not be stable".format(max_iter), RuntimeWarning)
            break
    return x

def plain_gd_arr(x, grad_func, ss, obj_func, max_iter, tol, **kwargs):
    """
    x0 must be either scalar or np.array
    """
    x0 = x + 2.0 * tol
    iter_count = 0

    while npla.norm(x - x0) > tol:
        np.copyto(x0, x)
        x = x0 - ss * grad_func(x0, **kwargs)
        iter_count += 1
        
        if iter_count > max_iter:
            warnings.warn("max_iter={0} is reached; result may not be stable".format(max_iter), RuntimeWarning)
            break
    return x

