import numpy as np
from sklearn.linear_model import Lasso

def lasso_skl(X, y, lam, abs_tol, iter_max):
    n = X.shape[0]
    las = Lasso(alpha=lam * 1.0 / n, tol=abs_tol, max_iter=iter_max, fit_intercept=False)
    las.fit(X, y)
    return las.coef_

def lasso_vec_skl(X, y, lams, abs_tol, iter_max):
    n, p = X.shape
    k = lams.shape[0]
    Beta_hat = np.zeros((p, k), dtype=np.float64)
    for i in range(k-1, -1, -1):
        las = Lasso(alpha=lams[i] * 1.0 / n, tol=abs_tol, max_iter=iter_max, fit_intercept=False)
        las.fit(X, y)
        Beta_hat[:, i] = las.coef_
    return Beta_hat

def lasso_skli(X, y, lam, abs_tol, iter_max):
    n, p = X.shape
    las = Lasso(alpha=lam * 1.0 / n, tol=abs_tol, max_iter=iter_max, fit_intercept=True)
    las.fit(X, y)
    beta_hat = np.zeros(p+1, dtype=np.float64)
    beta_hat[0] = las.intercept_
    beta_hat[1:] = las.coef_
    return beta_hat

def lasso_vec_skli(X, y, lams, abs_tol, iter_max):
    n, p = X.shape
    k = lams.shape[0]
    Beta_hat = np.zeros((p+1, k), dtype=np.float64)
    for i in range(k-1, -1, -1):
        las = Lasso(alpha=lams[i] * 1.0 / n, tol=abs_tol, max_iter=iter_max, fit_intercept=True)
        las.fit(X, y)
        Beta_hat[0, i] = las.intercept_
        Beta_hat[1:, i] = las.coef_
    return Beta_hat


