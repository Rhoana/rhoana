from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("fast_create_remap", ["fast_create_remap.pyx"])],
    include_dirs = [np.get_include()]
)