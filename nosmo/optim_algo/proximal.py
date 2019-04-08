import numpy as np

def L_inf_prox(z, tau):
    p = z.shape[0]
    zs = np.fabs(z)
    z_rank = np.argsort(zs)
    zs = np.sort(zs)

    xmax = zs[-1] - tau
    i = 1
    while xmax <= zs[-(i + 1)]:
        xmax = (xmax * i + zs[-(i + 1)]) / (i + 1.0)
        i += 1
        if i + 1 > p:
            break
    zs[(-i) : ] = xmax
    tmp = np.empty(p)
    tmp[z_rank] = zs
    return np.copysign(tmp, z)

def slope_prox(z, tau):
    return None

def proj_positive(z):
    return np.where(z >= 0, z, 0)

def proj_svm(z, y):
    cond = z + y / 2.0
    return np.where(np.fabs(cond) <= 0.5, z,  (- y + np.sign(cond)) / 2.0)
