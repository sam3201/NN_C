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
        sources=['src/c_modules/consciousness_final.c'],
        include_dirs=['src/c_modules', 'include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'orchestrator_and_agents', # New name for the combined extension
        sources=['src/c_modules/multi_agent_orchestrator_c.c', 'src/c_modules/specialized_agents_c.c'], # Combine source files
        include_dirs=['src/c_modules', 'include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=["-Wl,-export_dynamic"] # Add this line
    ),
        Extension(
            "sam_regulator_c",
            ["src/c_modules/sam_regulator_c.c"],
            include_dirs=["include"],
            extra_compile_args=["-O3"],
        ),
        Extension(
            "sam_meta_controller_c",
        sources=['src/c_modules/sam_meta_controller_c.c'],
        include_dirs=['src/c_modules', 'include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'sav_core_c',
        sources=['src/c_modules/sav_core_c.c'],
        include_dirs=['src/c_modules', 'include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'sam_sav_dual_system',
        sources=['src/c_modules/sam_sav_dual_system.c'],
        include_dirs=['src/c_modules', 'include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    # New C modules - Pure C implementations
    Extension(
        'sam_fast_rng',
        sources=['src/c_modules/sam_fast_rng.c'],
        include_dirs=['include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'sam_telemetry_core',
        sources=['src/c_modules/sam_telemetry_core.c'],
        include_dirs=['include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'sam_god_equation',
        sources=['src/c_modules/sam_god_equation.c'],
        include_dirs=['include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'sam_regulator_compiler_c',
        sources=['src/c_modules/sam_regulator_compiler.c', 'src/c_modules/sam_fast_rng.c'],
        include_dirs=['include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'sam_consciousness',
        sources=['src/c_modules/sam_consciousness.c'],
        include_dirs=['include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
    Extension(
        'sam_memory',
        sources=['src/c_modules/sam_memory.c'],
        include_dirs=['include'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=COMMON_ARGS,
        extra_link_args=[]
    ),
]

setup(
    name='SAM-C-Extensions',
    version='2.0.0',
    description='C extensions for SAM-D AGI system',
    ext_modules=extensions,
)
