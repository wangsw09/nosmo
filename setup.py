from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules_cprox = [
        Extension("nosmo.cprox.l_inf_prox", libraries=["m"], sources=["nosmo/cprox/l_inf_prox.pyx"]),
        Extension("nosmo.cprox.grouped_L1_prox", sources=["nosmo/cprox/grouped_L1_prox.pyx"]),
        Extension("nosmo.cprox.L1_proj", libraries=["m"], sources=["nosmo/cprox/L1_proj.pyx"]),
        Extension("nosmo.cprox.sorted_L1_prox", sources=["nosmo/cprox/sorted_L1_prox.pyx"]),
        Extension("nosmo.cprox.psd_proj", sources=["nosmo/cprox/psd_proj.pyx"]),
        Extension("nosmo.cprox.pquad_proj", sources=["nosmo/cprox/pquad_proj.pyx"]),
        Extension("nosmo.cprox.psd_proj_jacob", sources=["nosmo/cprox/psd_proj_jacob.pyx"]),
        Extension("nosmo.cprox.prox_wrapper", sources=["nosmo/cprox/prox_wrapper.pyx"])
        ]

ext_modules_cblas = [
        Extension("nosmo.clinalg.cython_blas_wrapper", sources=["nosmo/clinalg/cython_blas_wrapper.pyx"]),
        Extension("nosmo.clinalg.cython_lapack_wrapper", sources=["nosmo/clinalg/cython_lapack_wrapper.pyx"])
        ]

ext_modules_coptimization = [
        Extension("nosmo.coptimization.lasso_cd", sources=["nosmo/coptimization/lasso_cd.pyx"]),
        Extension("nosmo.coptimization.ridge_cd", sources=["nosmo/coptimization/ridge_cd.pyx"]),
        Extension("nosmo.coptimization.L_inf_pxgd", sources=["nosmo/coptimization/L_inf_pxgd.pyx"]),
        Extension("nosmo.coptimization.L_inf_arpxgd", sources=["nosmo/coptimization/L_inf_arpxgd.pyx"]),
        Extension("nosmo.coptimization.L_inf_admm", sources=["nosmo/coptimization/L_inf_admm.pyx"]),
        Extension("nosmo.coptimization.group_lasso_pxgd", sources=["nosmo/coptimization/group_lasso_pxgd.pyx"]),
        Extension("nosmo.coptimization.slope_pxgd", sources=["nosmo/coptimization/slope_pxgd.pyx"]),
        Extension("nosmo.coptimization.slope_arpxgd", sources=["nosmo/coptimization/slope_arpxgd.pyx"]),
        Extension("nosmo.coptimization.posv_ridge_pjgd", sources=["nosmo/coptimization/posv_ridge_pjgd.pyx"]),
        Extension("nosmo.coptimization.psd_matrix_ridge_pjgd", sources=["nosmo/coptimization/psd_matrix_ridge_pjgd.pyx"])
        ]

ext_modules_cCV = [
        Extension("nosmo.cross_validate.loocv", sources=["nosmo/cross_validate/loocv.pyx"])
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

