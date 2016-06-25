#!/usr/bin/env python3

from distutils.core import setup, Extension
from Cython.Build import cythonize

gibbsmodule = Extension(
    'gibbs',
    sources=['gibbs.c'],
    libraries=[],
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

setup(
    name = 'Gibbs aligner',
    ext_modules = cythonize("cyalign.pyx") + [gibbsmodule]
)

