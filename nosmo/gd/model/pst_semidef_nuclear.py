import numpy as np
from scipy.linalg import eigh as max_eigval
from .. import loss_reg as lr
from .. import proj
from .. import lib

def linear_pst_ridge(y, X, lam, max_iter=1000, tol=1e-7, algo="proj_gd"):
    """
    This function had not been finished;
    """
    n, p = X.shape
    x = np.zeros(p)
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    ss = 1.0 / (max_eigval(XTX, eigvals_only=True, eigvals=(p-1, p-1),
            check_finite=False)[0] + lam)

    ridge_obj = lambda x: lr.l2_loss_obj(x, y, X) + lam * lr.l2_reg_obj(x)
    ridge_grad = lambda x, XTy, XTX: lr.l2_loss_grad2(x, XTy, XTX) + lam * lr.l2_reg_grad(x)

    if algo == "proj_gd":
        x_hat = lib.proj_gd(x, ridge_grad, proj.pst_quadrant,
                ss, ridge_obj, max_iter, tol,
                grad={"XTX" : XTX, "XTy" : XTy}, proj={} )

    return x_hat

