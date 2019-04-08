import numpy as np
import numpy.random as npr
from scipy.linalg import eigh as eig_max
from sklearn import svm

from .proximal import *

def svm_linear(y, X, lam, intercept=False, tol=1e-8, max_iter=2000):
    clf = svm.LinearSVC(loss='hinge', C=1.0 / lam, tol=tol,
            fit_intercept=intercept, max_iter=max_iter)
    clf.fit(X, y)
    beta = clf.coef_[0]
    return beta

# def svm(y, X, lam, intercept=False, tol=1e-5, max_iter=500):
#     if intercept is False:
#         n, p = X.shape
#         XXT = np.dot(X, X.T)
#         ss = 0.9 * lam / eig_max(XXT, eigvals=(n-1, n-1), eigvals_only=True)[0]
#         mu0 = np.ones(n)
#         mu = np.zeros(n)
#         
#         iter_count = 0
#         
#         while np.amax(np.fabs(mu - mu0)) > tol and iter_count < max_iter:
#             np.copyto(mu0, mu)
#             mu = proj_svm(mu - ss / lam * np.dot(XXT, mu) - ss * y, y)
#             iter_count += 1
#         
#         print iter_count
#         beta = - np.dot(X.T, mu) / lam
#         return beta
# 
#     else:
#         return None
# 
def svm_ker(y, X, lam, intercept=False):
    return None
