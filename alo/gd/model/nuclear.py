import numpy as np
import numpy.linalg as npla
from scipy.linalg import eigh as max_eigval
from .. import loss_reg as lr
from .. import lib

def linear_nuclear(y, X, lam, max_iter=1000, tol=1e-7, algo="prox_gd"):
    n, p1, p2 = X.shape
    x = np.zeros(shape=(p1, p2))
    # XTX = np.dot(X.T, X)
    # XTy = np.dot(X.T, y)
    ss = 1.0 / (npla.svd(X.reshape(n, -1), full_matrices=False,
            compute_uv=False)[0] ** 2.0)

    nuclear_obj = lambda x: lr.mat_l2_loss_obj(x, y, X) + lam * lr.nuclear_reg_obj(x)

    if algo == "prox_gd":
        x_hat = lib.prox_gd(x, lr.mat_l2_loss_grad1, lr.nuclear_reg_prox,
                ss, lam, nuclear_obj, max_iter, tol,
                grad={"y" : y, "X" : X}, prox={} )
    elif algo == "acc_prox_gd":
        x_hat = lib.acc_prox_gd(x, lr.mat_l2_loss_grad1, lr.nuclear_reg_prox,
                ss, lam, nuclear_obj, max_iter, tol,
                grad={"y" : y, "X" : X}, prox={} )

    return x_hat

def logistic_lasso():
    pass

def poisson_lasso():
    pass

