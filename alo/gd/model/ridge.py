import numpy as np
from scipy.linalg import eigh as max_eigval
from .. import loss_reg as lr
from .. import lib

def linear_ridge(y, X, lam, max_iter=1000, tol=1e-7, algo="gd"):
    n, p = X.shape
    x = np.zeros(p)
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    ss = 1.0 / (max_eigval(XTX, eigvals_only=True, eigvals=(p-1, p-1),
            check_finite=False)[0] + lam)

    ridge_obj = lambda x: lr.l2_loss_obj(x, y, X) + lam * lr.l2_reg_obj(x)
    ridge_grad = lambda x, XTy, XTX: lr.l2_loss_grad2(x, XTy, XTX) + lam * lr.l2_reg_grad(x)
    
    if algo == "gd":
        x_hat = lib.plain_gd(x, ridge_grad,
                ss, ridge_obj, max_iter, tol, XTy=XTy, XTX=XTX )
    elif algo == "acc_gd":
        x_hat = lib.acc_gd(x, ridge_grad,
                ss, ridge_obj, max_iter, tol, XTy=XTy, XTX=XTX )
    elif algo == "prox_gd":
        x_hat = lib.prox_gd(x, lr.l2_loss_grad2, lr.l2_reg_prox,
                ss, lam, ridge_obj, max_iter, tol,
                grad={"XTy" : XTy, "XTX" : XTX}, prox={} )
    elif algo == "acc_prox_gd":
        x_hat = lib.acc_prox_gd(x, lr.l2_loss_grad2, lr.l2_reg_prox,
                ss, lam, ridge_obj, max_iter, tol,
                grad={"XTy" : XTy, "XTX" : XTX}, prox={} )
    elif algo == "acc_gd_restart":
        x_hat = lib.acc_gd_restart(x, ridge_grad,
                ss, ridge_obj, max_iter, tol, XTy=XTy, XTX=XTX )
    elif algo == "acc_prox_gd_restart":
        x_hat = lib.acc_prox_gd_restart(x, lr.l2_loss_grad2, lr.l2_reg_prox,
                ss, lam, ridge_obj, max_iter, tol,
                grad={"XTy" : XTy, "XTX" : XTX}, prox={} )

    return x_hat

def logistic_ridge():
    pass

def poisson_ridge():
    pass


