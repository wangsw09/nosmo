import numpy as np
from sklearn.svm import LinearSVC as lmsvm

def svm_skl(X, y, lam, abs_tol, iter_max):
    svc = lmsvm(penalty="l2", loss="hinge", tol=abs_tol, C=1.0 / lam,
            max_iter=iter_max, fit_intercept=False)
    svc.fit(X, y)
    return svc.coef_[0]

def svm_vec_skl(X, y, lams, abs_tol, iter_max):
    n, p = X.shape
    k = lams.shape[0]
    Beta_hat = np.zeros((p, k), dtype=np.float64)
    for i in range(k-1, -1, -1):
        svc = lmsvm(penalty="l2", loss="hinge", tol=abs_tol, C=1.0 / lams[i],
                max_iter=iter_max, fit_intercept=False)
        svc.fit(X, y)
        Beta_hat[:, i] = svc.coef_[0]
    return Beta_hat

def svm_skli(X, y, lam, abs_tol, iter_max):
    p = X.shape[1]
    svc = lmsvm(penalty="l2", loss="hinge", tol=abs_tol, C=1.0 / lam,
            max_iter=iter_max, fit_intercept=True)
    beta_hat = np.zeros(p + 1, dtype=np.float64)
    svc.fit(X, y)
    beta_hat[0] = svc.intercept_[0]
    beta_hat[1:] = svc.coef_[0]
    return beta_hat

def svm_vec_skli(X, y, lams, abs_tol, iter_max):
    n, p = X.shape
    k = lams.shape[0]
    Beta_hat = np.zeros((p + 1, k), dtype=np.float64)
    for i in range(k-1, -1, -1):
        svc = lmsvm(penalty="l2", loss="hinge", tol=abs_tol, C=1.0 / lams[i],
                max_iter=iter_max, fit_intercept=True)
        svc.fit(X, y)
        Beta_hat[0, i] = svc.intercept_[0]
        Beta_hat[1:, i] = svc.coef_[0]
    return Beta_hat


