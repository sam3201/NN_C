#!/usr/bin/env python3
"""
Setup script for SAM Survival C Library
Compiles the performance-critical survival and goal management functions
"""

from setuptools import setup, Extension
import sys
import os

# Define the C extension module
survival_module = Extension(
    'sam_survival_c',
    sources=['sam_survival_c.c'],
    include_dirs=[],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[
        '-O3',  # Maximum optimization
        '-march=native',  # Use native CPU features
        '-ffast-math',  # Fast math operations
        '-funroll-loops',  # Unroll loops for speed
    ] if sys.platform != 'win32' else [
        '/O2',  # Maximum optimization for Windows
    ],
    extra_link_args=[],
    define_macros=[
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
    ]
)

setup(
    name='sam-survival-c',
    version='2.0.0',
    description='SAM Survival and Goal Management C Library',
    author='SAM System',
    ext_modules=[survival_module],
    python_requires='>=3.6',
)
