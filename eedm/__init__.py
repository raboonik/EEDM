'''
    EEDM: EigenEnergy Decomposition Method
    
    Main package initializer. Exposing the core functions of the package
    
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
'''

import importlib.util
import os
import sys


def _load_settings():
    '''
        Make the user's settings.py importable as "settings" before any EEDM module needs it. It is found as a normal
        module (e.g. in the directory "python -m eedm" is run from), or else in the current directory, which also covers
        the "eedm" command, whose import path does not include the current directory.
    '''
    try:
        import settings  # noqa: F401
        return
    except ModuleNotFoundError as e:
        if e.name != "settings": raise
    path = os.path.join(os.getcwd(), "settings.py")
    if not os.path.isfile(path):
        raise ModuleNotFoundError("EEDM reads its settings from a settings.py in the directory it is run from, but there is none in " + os.getcwd() +
                                  ". Run EEDM from the directory holding your settings.py (e.g. EEDM/), or copy EEDM/settings.py there.", name="settings")
    spec   = importlib.util.spec_from_file_location("settings", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["settings"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules["settings"]
        raise


_load_settings()

# Core submodules
from . import AutoParallelizePy as APP
from . import io
from . import methods
from . import utils

from .io import larexd
from .io import mancha
from .io import pencil

# Common utilities
from . import context
from . import system
from . import SI_constants as const
from . import decorators

# Define public API for external users
__all__ = [
    "APP",
    "io", "larexd", "mancha", "pencil",
    "methods", 
    "utils",
    "context", "system", "const", "decorators",
]