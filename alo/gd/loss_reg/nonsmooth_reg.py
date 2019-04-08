import numpy as np
import numpy.linalg as npla

# L1
def l1_reg_obj(x):
    return np.sum(np.absolute(x))

def l1_reg_grad(x):
    return np.sign(x)

def l1_reg_prox(x, tau, **kwargs):
    s = np.sign(x)
    a = np.absolute(x)
    return np.where(a > tau, s * (a - tau), 0)


# nuclear norm
def nuclear_reg_obj(x):
    return npla.norm(x, ord="nuc")

# def nuclear_reg_grad(x):
#     return np.sign(x)

def nuclear_reg_prox(x, tau, **kwargs):
    U, s, VT = npla.svd(x, full_matrices=False, compute_uv=True)
    s_thresh = np.where(s > tau, s - tau, 0.0)
    return np.dot(U * s_thresh, VT)


# L_inf
def L_inf_reg_obj(x):
    return np.amax(np.fabs(x))

def L_inf_reg_prox(x, tau, **kwargs):
    p = x.shape[0]
    xs = np.fabs(x)
    x_rank = np.argsort(xs)
    xs = np.sort(xs)

    xmax = xs[-1] - tau
    i = 1
    while xmax <= xs[-(i + 1)]:
        xmax = (xmax * i + xs[-(i + 1)]) / (i + 1.0)
        i += 1
        if i + 1 > p:
            break
    xs[(-i) : ] = xmax
    tmp = np.empty(p)
    tmp[x_rank] = xs
    return np.copysign(tmp, x)


def slope_reg_obj(x, theta):
    """
    theta should be in decreasing order
    """
    return np.dot(np.sort(np.fabs(x))[::-1], theta)

def slope_reg_prox(x, tau, theta, **kwargs):
    """
    theta should be in decreasing order
    """
    p = x.shape[0]
    xs = np.fabs(x)
    x_rank = np.argsort(xs)
    xs = np.sort(xs)[::-1] - tau * theta
    
    stack = [(0, 0, xs[0], xs[0] if xs[0] > 0 else 0.0)]
    for i in xrange(1, p):
        stack.append((i, i, xs[i], pst_part(xs[i])))
        while len(stack) > 1 and stack[-1][3] >= stack[-2][3]:
            tmp = stack.pop()
            stack[-1] = (stack[-1][0], tmp[1], stack[-1][2] + tmp[2],
                    pst_part((stack[-1][2] + tmp[2]) / (tmp[1] - stack[-1][0]
                        + 1.0)))

    for block in stack:
        xs[block[0] : block[1] + 1] = block[3]

    tmp_x = np.empty(p)
    tmp_x[x_rank] = xs[::-1]
    return np.copysign(tmp_x, x)

def pst_part(x):
    return x if x > 0.0 else 0.0

def ngt_part(x):
    return x if x < 0.0 else 0.0
