import numpy as np
import numpy.linalg as npla

def l2_ball(x, r):
    return x / npla.norm(x) * r

def l1_ball(x, r):
    pass
