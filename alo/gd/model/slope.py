import numpy as np
from scipy.linalg import eigh as max_eigval
from .. import loss_reg as lr
from .. import lib

def linear_slope(y, X, lam, theta, max_iter=1000, tol=1e-7, algo="prox_gd"):
    """
    theta in decreasing order
    """
    n, p = X.shape
    x = np.zeros(p)
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    ss = 1.0 / max_eigval(XTX, eigvals_only=True, eigvals=(p-1, p-1),
            check_finite=False)[0]

    slope_obj = lambda x: lr.l2_loss_obj(x, y, X) + lam * lr.slope_reg_obj(x,
            theta)

    if algo == "prox_gd":
        x_hat = lib.prox_gd(x, lr.l2_loss_grad2, lr.slope_reg_prox,
                ss, lam, slope_obj, max_iter, tol,
                grad={"XTy" : XTy, "XTX" : XTX}, prox={"theta" : theta} )
    elif algo == "acc_prox_gd":
        x_hat = lib.acc_prox_gd(x, lr.l2_loss_grad2, lr.slope_reg_prox,
                ss, lam, slope_obj, max_iter, tol,
                grad={"XTy" : XTy, "XTX" : XTX}, prox={"theta" : theta} )
    elif algo == "acc_prox_gd_restart":
        x_hat = lib.acc_prox_gd_restart(x, lr.l2_loss_grad2, lr.slope_reg_prox,
                ss, lam, slope_obj, max_iter, tol,
                grad={"XTy" : XTy, "XTX" : XTX}, prox={"theta" : theta} )

    return x_hat

def logistic_lasso():
    pass

def poisson_lasso():
    pass

