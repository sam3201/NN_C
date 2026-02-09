from setuptools import setup, Extension

# Define the C extensions (only include ones with fixed includes)
extensions = [
    Extension(
        'consciousness_algorithmic',
        sources=['consciousness_algorithmic.c'],
        include_dirs=['.'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=['-O3', '-march=native'],
        extra_link_args=[]
    ),
    Extension(
        'multi_agent_orchestrator_c',
        sources=['multi_agent_orchestrator_c.c'],
        include_dirs=['.'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=['-O3', '-march=native'],
        extra_link_args=[]
    ),
    Extension(
        'specialized_agents_c',
        sources=['specialized_agents_c.c'],
        include_dirs=['.'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=['-O3', '-march=native'],
        extra_link_args=[]
    ),
    Extension(
        'sam_meta_controller_c',
        sources=['sam_meta_controller_c.c'],
        include_dirs=['.'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=['-O3', '-march=native'],
        extra_link_args=[]
    ),
    Extension(
        'sav_core_c',
        sources=['sav_core_c.c'],
        include_dirs=['.'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=['-O3', '-march=native'],
        extra_link_args=[]
    ),
    Extension(
        'sam_sav_dual_system',
        sources=['sam_sav_dual_system.c'],
        include_dirs=['.'],
        library_dirs=[],
        libraries=[],
        extra_compile_args=['-O3', '-march=native'],
        extra_link_args=[]
    )
]

setup(
    name='SAM-C-Extensions',
    version='2.0.0',
    description='C extensions for SAM 2.0 AGI system',
    ext_modules=extensions,
)
