#!/usr/bin/env python3
"""
Setup script for SAM C extensions
Builds all C modules as Python extensions
"""

from setuptools import setup, Extension
import os

# Include paths for SAM framework
include_dirs = [
    './ORGANIZED/UTILS/SAM/SAM',
    './ORGANIZED/UTILS/utils/NN/MUZE',
    './ORGANIZED/UTILS/utils/NN/NEAT',
    './ORGANIZED/UTILS/utils/NN/TRANSFORMER'
]

# Consciousness module extension
consciousness_ext = Extension(
    'consciousness_algorithmic',
    sources=['consciousness_algorithmic.c'],
    include_dirs=include_dirs,
    extra_compile_args=['-std=c99', '-O2', '-Wall', '-Wextra'],
    libraries=['m']
)

# Multi-agent orchestrator extension
orchestrator_ext = Extension(
    'multi_agent_orchestrator_c',
    sources=['multi_agent_orchestrator_c.c', './ORGANIZED/UTILS/SAM/SAM/SAM.c', './ORGANIZED/UTILS/utils/NN/NEAT/NEAT.c', './ORGANIZED/UTILS/utils/NN/NN/NN.c', './ORGANIZED/UTILS/utils/NN/TRANSFORMER/TRANSFORMER.c'],
    include_dirs=include_dirs,
    extra_compile_args=['-std=c99', '-O2', '-Wall', '-Wextra'],
    libraries=['m', 'pthread']
)

# Specialized agents extension
agents_ext = Extension(
    'specialized_agents_c',
    sources=['specialized_agents_c.c'],
    include_dirs=include_dirs,
    extra_compile_args=['-std=c99', '-O2', '-Wall', '-Wextra'],
    libraries=['m', 'pthread']
)

# Neural network extension - REMOVED: using existing NN library
# neural_ext = Extension(
#     'neural_network_c',
#     sources=['neural_network_c.c'],
#     include_dirs=include_dirs,
#     extra_compile_args=['-std=c99', '-O2', '-Wall', '-Wextra'],
#     libraries=['m']
# )

# Web server extension - REMOVED: using existing Python web server
# webserver_ext = Extension(
#     'sam_web_server_c',
#     sources=['sam_web_server_c.c'],
#     include_dirs=include_dirs,
#     extra_compile_args=['-std=c99', '-O2', '-Wall', '-Wextra'],
#     libraries=['m', 'pthread']
# )

setup(
    name='sam-c-extensions',
    version='2.0.0',
    description='Pure C implementations for SAM AGI system',
    author='SAM AGI',
    author_email='sam@agi.system',
    ext_modules=[
        consciousness_ext,
        orchestrator_ext,
        agents_ext
    ],
    python_requires='>=3.8'
)
