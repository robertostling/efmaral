#!/usr/bin/env python3

from distutils.core import setup, Extension
from Cython.Build import cythonize

gibbsmodule = Extension(
    'gibbs',
    sources=['gibbs.c'],
    libraries=[],
    extra_compile_args=['-std=c99', '-Wall', '-fopenmp'],
    extra_link_args=['-lgomp'])

setup(
    name = 'Gibbs aligner',
    ext_modules = cythonize("cyalign.pyx") + [gibbsmodule]
)

