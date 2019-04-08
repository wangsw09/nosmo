import numpy as np

def l2_loss_obj(x, y, X):
    return 0.5 * np.sum((y - np.dot(X, x)) ** 2)

def l2_loss_grad1(x, y, X):
    return np.dot(X.T, np.dot(X, x) - y)

def l2_loss_grad2(x, XTy, XTX):
    return np.dot(XTX, x) - XTy


def mat_l2_loss_obj(x, y, X):
    """
    X is 3-dim array with shape (n, p1, p2)
    x is 2-dim array with shape (p1, p2)
    y is 10dim array with shape (n)
    """
    return 0.5 * np.sum((y - np.sum(X * x, axis=(1, 2)) ) ** 2)

def mat_l2_loss_grad1(x, y, X):
    return np.sum(X * (np.sum(X * x, axis=(1, 2)) - y)[:, np.newaxis,
        np.newaxis], axis = 0)

# def mat_l2_loss_grad2(x, XTy, XTX):
#     return np.dot(XTX, x) - XTy



def svm_dual_l2_obj(alpha, XDy, lam):
    return np.sum(np.dot(XDy.T, alpha) ** 2) / 2.0 / lam - np.sum(alpha)

def svm_dual_l2_grad(alpha, XDy, lam):
    return np.dot(XDy, np.dot(XDy.T, alpha)) / lam - 1.0

