import numpy as np
from scipy.linalg import eigh as max_eigval
from .. import loss_reg as lr
from .. import proj
from .. import lib

def linear_svm(y, X, lam, max_iter=1000, tol=1e-7, algo="proj_gd"):
    n, p = X.shape
    alpha = np.zeros(n)
    XTX = np.dot(X.T, X)
    XDy = X * y[:, np.newaxis]
    ss = lam / max_eigval(XTX, eigvals_only=True, eigvals=(p-1, p-1),
            check_finite=False)[0]

    svm_obj = lambda a: lr.svm_dual_l2_obj(a, XDy, lam)

    if algo == "proj_gd":
        alpha_hat = lib.proj_gd(alpha, lr.svm_dual_l2_grad, proj.interval,
                ss, svm_obj, max_iter, tol,
                grad={"XDy" : XDy, "lam" : lam},
                proj={"a" : 0.0, "b" : 1.0} )
    elif algo == "acc_proj_gd":
        alpha_hat = lib.acc_proj_gd(alpha, lr.svm_dual_l2_grad, proj.interval,
                ss, svm_obj, max_iter, tol,
                grad={"XDy" : XDy, "lam" : lam},
                proj={"a" : 0.0, "b" : 1.0} )
    elif algo == "acc_proj_gd_restart":
        alpha_hat = lib.acc_proj_gd_restart(alpha, lr.svm_dual_l2_grad, proj.interval,
                ss, svm_obj, max_iter, tol,
                grad={"XDy" : XDy, "lam" : lam},
                proj={"a" : 0.0, "b" : 1.0} )
    
    E = (alpha_hat > 0.0)
    return np.dot(XDy[E, :].T, alpha_hat[E]) / lam

