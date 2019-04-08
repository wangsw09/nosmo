import sys
import os
import time
import subprocess
import re

import numpy as np
import numpy.random as npr

from ..coptimization import L_inf_pxgd, L_inf_vec_pxgd, group_lasso_pxgd, group_lasso_vec_pxgd, slope_pxgd, slope_vec_pxgd

MAXARRAYSIZE=1000
TMPFILEFOLDER = "TMPABCDFILES"
TMPFILELOC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), TMPFILEFOLDER)
PYSCRIPTNAME = "TMP_PY_SCRIPT.py"
SHSCRIPTNAME = "TMP_SH_SCRIPT.sh"
MERGESCRIPTNAME = "TMP_MERGE_SCRIPT.py"
MERGESHSCRIPTNAME = "TMP_MERGE_SH_SCRIPT.sh"
RETARRAYNAME = "TMP_PRED_ARRAY"
LOOERRNAME = "loocv_err.npy"

def generate_py_script(TMP_FEATURE_DATA):
    text = u"""import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import alo

def run_loocv(i):
    data = np.load('{tfd}')
    X, y, lams, abs_tol, iter_max, beta_full = data['X'], data['y'], data['lams'], data['abs_tol'], data['iter_max'], data['beta_full']
    n, p = X.shape

    mask = np.ones(n, dtype=np.bool)
    mask[i] = False

    Beta_hat = alo.coptimization.L_inf_vec_pxgd(X[mask, :], y[mask], beta_full, lams, abs_tol, iter_max)

    return np.dot(Beta_hat.T, X[i, :])

if __name__ == '__main__':
    i = int(sys.argv[1])
    pred_i = run_loocv(i)
    np.save('{tpa}_{{i}}.npy'.format(i=i), pred_i)
    """.format(tfd=os.path.join(TMPFILELOC, TMP_FEATURE_DATA), tpa=os.path.join(TMPFILELOC, RETARRAYNAME))
    with open(os.path.join(TMPFILELOC, PYSCRIPTNAME), "w") as fp:
        fp.write(text)

def generate_merge_py_script(TMP_FEATURE_DATA):
    text = """import numpy as np

data = np.load('{tfd}')
X, y, lams, abs_tol, iter_max, beta_full = data['X'], data['y'], data['lams'], data['abs_tol'], data['iter_max'], data['beta_full']
n = X.shape[0]
lam_len = lams.shape[0]
Pred = np.zeros((n, lam_len), dtype=np.float64)
for i in range(n):
    Pred[i, :] = np.load('{tpa}_{{i}}.npy'.format(i=i))
loocv_err = np.mean((y[:, np.newaxis] - Pred) ** 2, axis=0)

if __name__ == '__main__':
    np.save('{loo}', loocv_err)
    """.format(tfd=os.path.join(TMPFILELOC, TMP_FEATURE_DATA),
               tpa=os.path.join(TMPFILELOC, RETARRAYNAME),
               loo=os.path.join(TMPFILELOC, LOOERRNAME))
    with open(os.path.join(TMPFILELOC, MERGESCRIPTNAME), "w") as fp:
        fp.write(text)

def generate_sh_script(n):
    text = """#!/bin/bash
#SBATCH --account=stats
#SBATCH --output={out}
#SBATCH --job-name=loocv
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 2G
#SBATCH --time 01:00:00

python {psn} $SLURM_ARRAY_TASK_ID
    """.format(psn=os.path.join(TMPFILELOC, PYSCRIPTNAME), out=os.path.join(TMPFILELOC, "slurm_job_\%A_\%a.out"))
    with open(os.path.join(TMPFILELOC, SHSCRIPTNAME), "w") as fp:
        fp.write(text)

def generate_merge_sh_script():
    text = """#!/bin/bash
#SBATCH --account=stats
#SBATCH --output={out}
#SBATCH --job-name=loocv
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 2G
#SBATCH --time 00:03:00

python {psn}
    """.format(psn=os.path.join(TMPFILELOC, MERGESCRIPTNAME), out=os.path.join(TMPFILELOC, "slurm_job_\%A_\%a.out"))
    with open(os.path.join(TMPFILELOC, MERGESHSCRIPTNAME), "w") as fp:
        fp.write(text)

def generate_data(n, p, k, lam_len, lam_start, lam_end):
    npr.seed(124)
    X = npr.normal(size=(n, p)) / np.sqrt(n)
    beta = np.zeros(p, dtype=np.float64)
    mask = np.ones(p, dtype=np.bool)
    mask[npr.choice(p, k, replace=False)] = False
    beta[mask] = npr.uniform(-3, 3, size=p-k)
    beta[~mask] = npr.choice([-3, 3], size=k, replace=True)
    
    noise = npr.normal(size=n) * 0.8
    y = np.dot(X, beta) + noise

    lams = 10 ** np.linspace(lam_start, lam_end, lam_len)
    return X, y, lams

def main(X, y, lams, abs_tol, iter_max):
    # n = 20
    # p = 40
    # k = 5
    # lam_len = 20
    # lam_start = 2
    # lam_end = 4
    # abs_tol = 1e-10
    # iter_max = 100000
    n, p = X.shape
    TMP_FEATURE_DATA = "TMP_feature_data_n{n}_p{p}.npz".format(n=n, p=p)
    os.mkdir(TMPFILELOC)

    # X, y, lams = generate_data(n, p, k, lam_len, lam_start, lam_end)
    beta_full = L_inf_pxgd(X, y, np.zeros(p, dtype=np.float64), lams[-1], abs_tol, iter_max)
    np.savez(os.path.join(TMPFILELOC, TMP_FEATURE_DATA),
            X=X, y=y, lams=lams, beta_full=beta_full, abs_tol=abs_tol, iter_max=iter_max)

    generate_py_script(TMP_FEATURE_DATA)
    generate_sh_script(n)
    generate_merge_py_script(TMP_FEATURE_DATA)
    generate_merge_sh_script()

    job_arr_id = subprocess.check_output(["sbatch", "--array=0-{num}".format(num=n-1), os.path.join(TMPFILELOC, SHSCRIPTNAME)])
    job_arr_id = re.search(r"(\d+)", job_arr_id).group()
    subprocess.check_output(["sbatch", "--depend=afterok:{id}".format(id=job_arr_id), os.path.join(TMPFILELOC, MERGESHSCRIPTNAME)])
    while not os.path.isfile(os.path.join(TMPFILELOC, LOOERRNAME)):
        time.sleep(0.5)

    loo_err = np.load(os.path.join(TMPFILELOC, LOOERRNAME))
    subprocess.call(["rm", "-r", TMPFILELOC])
    return loo_err

if __name__ == "__main__":
    main()

