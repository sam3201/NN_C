from __future__ import annotations

import os
import sys
from setuptools import setup, Extension

def _is_msvc() -> bool:
    return sys.platform.startswith('win')

def _common_compile_args() -> list[str]:
    if _is_msvc():
        return ['/O2']
    args = ['-O3']
    if os.environ.get('SAM_NATIVE') == '1':
        args.append('-march=native')
    return args

COMMON_ARGS = _common_compile_args()


# Define the C extensions (only include ones with fixed includes)
extensions = [
    Extension(
        'consciousness_algorithmic',
        sources=['src/c_modules/consciousness_algorithmic.c'],
        include_dirs=['src/c_modules'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'orchestrator_and_agents', # New name for the combined extension
        sources=['src/c_modules/multi_agent_orchestrator_c.c', 'src/c_modules/specialized_agents_c.c'], # Combine source files
        include_dirs=['src/c_modules'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=["-Wl,-export_dynamic"] # Add this line
    ),
    Extension(
        'sam_meta_controller_c',
        sources=['src/c_modules/sam_meta_controller_c.c'],
        include_dirs=['src/c_modules'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'sav_core_c',
        sources=['src/c_modules/sav_core_c.c'],
        include_dirs=['src/c_modules'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'sam_sav_dual_system',
        sources=['src/c_modules/sam_sav_dual_system.c'],
        include_dirs=['src/c_modules'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    )
]

setup(
    name='SAM-C-Extensions',
    version='2.0.0',
    description='C extensions for SAM 2.0 AGI system',
    ext_modules=extensions,
)
