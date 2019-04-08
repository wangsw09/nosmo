import numpy as np
import numpy.random as npr
import glmnet_python
from glmnet import glmnet

def glm_ridge(y, X, lam=None, lam_seq=None, intercept=False, family="gaussian"):
    """
    We are going to simply wrap the functions in glmnet.
    """
    n, p = X.shape
    if lam is not None:
        if not intercept:
            beta = glmnet(x=X, y=y / np.std(y), family=family, intr=False, alpha=0, nlambda=1,
                    lambdau=np.array([lam / n]), standardize=False)['beta'][:,
                            0] * np.std(y)
            return beta
        else:
            res = glmnet(x=X, y=y, family=family, intr=True, alpha=0, nlambda=1,
                    lambdau=np.array([lam / n]), standardize=False)
            return res['beta'][:, 0], res['a0'][0]
    elif lam_seq is not None:
        m = lam_seq.shape[0]
        if not intercept:
            beta = np.empty((p, m))
            lam_seq_sorted = np.sort(lam_seq)[::-1]
            lam_seq_rank = np.argsort(lam_seq)[::-1]
            if family == "gaussian":
                beta[:, lam_seq_rank] = glmnet(x=X, y=y / np.std(y), family=family, intr=False,
                        alpha=0, nlambda=1, lambdau=np.asarray(lam_seq_sorted) / n,
                        standardize=False)['beta'] * np.std(y)
            else:
                beta[:, lam_seq_rank] = glmnet(x=X, y=y, family=family, intr=False, alpha=0, nlambda=1,
                        lambdau=np.asarray(lam_seq_sorted) / n, standardize=False)['beta']
            return beta
        else:
            beta = np.empty((p, m))
            beta0 = np.empty(m)
            lam_seq_sorted = np.sort(lam_seq)[::-1]
            lam_seq_rank = np.argsort(lam_seq)[::-1]

            res = glmnet(x=X, y=y, family=family, intr=True, alpha=0, nlambda=1,
                    lambdau=np.asarray(lam_seq_sorted) / n, standardize=False)
            beta[:, lam_seq_rank] = res["beta"]
            beta0[lam_seq_rank] = res["a0"]
            return beta, beta0
 


