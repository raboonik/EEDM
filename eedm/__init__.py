'''
    EEDM: EigenEnergy Decomposition Method
    
    Main package initializer. Exposing the core functions of the package
    
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
'''

# Core submodules
from . import AutoParallelizePy as APP
from . import io
from . import methods
from . import utils

from .io import larexd

# Common utilities
from . import system
from . import SI_constants as const

# Define public API for external users
__all__ = [
    "APP",
    "io", "larexd",
    "methods", 
    "utils",
    "system", "const",
]