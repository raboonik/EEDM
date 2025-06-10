import h5py   as     h5
import numpy  as     np
import scipy  as     spy
import mpi4py
import glob, os, subprocess, sys   # subprocess for storing output of bash commands
from os import system as bash
from os import chdir as cd


# from random import *

def system_call(command):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if len(err) > 5:
        return 0
    else:
        return 1

def fid(path):
    return sorted(glob.glob(path))

def add_slash(path):
    if path[-1] != '/': path = path + '/'
    return path

from SI_constants import *
from settings import *

datapath = add_slash(datapath)

# EEDM Specific functions
from eigenDecompFuncs  import *
from differentiation   import *
from get_energy_sliced import *
from reader            import *

packages = ['libs/AutoParallelizePy/libs', 'libs/larexd']
for package in packages: sys.path.insert(0, os.path.join(os.getcwd(), package))


import AutoParallelizePy as APP

__all__ = ['np', 'spy', 'mpi4py','h5', 'bash', 'cd', 'system_call', 
           'fid', 'add_slash', 'APP']#,'constants', 'settings'

if dataExt == "sdf":
    try:
        import sdf
        __all__ .append('sdf')
    except:
        raise ValueError("SDF module not found!")
elif dataExt == "cfd":
    try:
        import read_lare_cfd3d as cfd3d
        import read_lare_cfd2d as cfd2d
        __all__ .append('cfd3d')
        __all__ .append('cfd2d')
    except:
        raise ValueError("CFD module not found!")


