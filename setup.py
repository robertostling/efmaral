#!/usr/bin/env python3

from setuptools import setup, Extension
#from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

gibbsmodule = Extension(
    'gibbs',
    sources=['gibbs.c'],
    libraries=[],
    include_dirs=[numpy.get_include()],
    # NOTE: the -Wno.. arguments are to compensate for a bug in the build
    # system
    extra_compile_args=['-std=c99', '-Wall', '-fopenmp',
                        '-Wno-error=declaration-after-statement',
                        '-Wno-declaration-after-statement',
                        '-Wno-unused-function',
                        # Enable this to use simd_math_primes.h for expf/logf
                        # This should speed up fertility distribution sampling
                        # by about 20%, so overall impact is quite limited.
                        #'-DAPPROXIMATE_MATH',
                        ],
    extra_link_args=['-lgomp'])

cyalign_ext=Extension('cyalign',['cyalign.pyx'],
                      include_dirs=[numpy.get_include()])

setup(
    name='efmaral',
    version='0.1',
    author='Robert Östling',
    url='https://github.com/robertostling/efmaral',
    license='GNU GPLv3',
    install_requires=['numpy'],
    ext_modules=cythonize(cyalign_ext) + [gibbsmodule]
)

