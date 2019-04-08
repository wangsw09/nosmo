import numpy as np

def pst_quadrant(x, **kwargs):
    return np.where(x > 0, x, 0)

def interval(x, a, b):
    return np.clip(x, a_min=a, a_max=b)

