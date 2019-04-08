from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules_cprox = [
        Extension("alo.cprox.l_inf_prox", libraries=["m"], sources=["alo/cprox/l_inf_prox.pyx"]),
        Extension("alo.cprox.grouped_L1_prox", sources=["alo/cprox/grouped_L1_prox.pyx"]),
        Extension("alo.cprox.L1_proj", libraries=["m"], sources=["alo/cprox/L1_proj.pyx"]),
        Extension("alo.cprox.sorted_L1_prox", sources=["alo/cprox/sorted_L1_prox.pyx"]),
        Extension("alo.cprox.psd_proj", sources=["alo/cprox/psd_proj.pyx"]),
        Extension("alo.cprox.pquad_proj", sources=["alo/cprox/pquad_proj.pyx"]),
        Extension("alo.cprox.psd_proj_jacob", sources=["alo/cprox/psd_proj_jacob.pyx"])
        ]

ext_modules_cblas = [
        Extension("alo.clinalg.cython_blas_wrapper", sources=["alo/clinalg/cython_blas_wrapper.pyx"]),
        Extension("alo.clinalg.cython_lapack_wrapper", sources=["alo/clinalg/cython_lapack_wrapper.pyx"])
        ]

ext_modules_coptimization = [
        Extension("alo.coptimization.lasso_cd", sources=["alo/coptimization/lasso_cd.pyx"]),
        Extension("alo.coptimization.ridge_cd", sources=["alo/coptimization/ridge_cd.pyx"]),
        Extension("alo.coptimization.L_inf_pxgd", sources=["alo/coptimization/L_inf_pxgd.pyx"]),
        Extension("alo.coptimization.L_inf_arpxgd", sources=["alo/coptimization/L_inf_arpxgd.pyx"]),
        Extension("alo.coptimization.L_inf_admm", sources=["alo/coptimization/L_inf_admm.pyx"]),
        Extension("alo.coptimization.group_lasso_pxgd", sources=["alo/coptimization/group_lasso_pxgd.pyx"]),
        Extension("alo.coptimization.slope_pxgd", sources=["alo/coptimization/slope_pxgd.pyx"]),
        Extension("alo.coptimization.slope_arpxgd", sources=["alo/coptimization/slope_arpxgd.pyx"]),
        Extension("alo.coptimization.posv_ridge_pjgd", sources=["alo/coptimization/posv_ridge_pjgd.pyx"]),
        Extension("alo.coptimization.psd_matrix_ridge_pjgd", sources=["alo/coptimization/psd_matrix_ridge_pjgd.pyx"])
        ]

ext_modules_cCV = [
        Extension("alo.cross_validate.loocv", sources=["alo/cross_validate/loocv.pyx"])
        ]

setup(
        name = "cAccelerate",
        cmdclass = {"build_ext" : build_ext},
        ext_modules = (
            ext_modules_cprox +
            ext_modules_cblas +
            ext_modules_coptimization +
            ext_modules_cCV),
        script_args=["build_ext"],
        options={"build_ext" : {"inplace" : True, "force" : True}},
        )

