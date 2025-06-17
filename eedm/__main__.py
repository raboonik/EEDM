'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    Date  : 2025-4-03
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Parallelized routine to work out the eigenenergy time derivatives (Eq6 of the paper) 
    and the eigenenergies themselves (Eq9).
    
    It works for full nonlinear ideal-MHD with constant gravity along the -z axis.
    
    Code structure:
        EEDM/
            ├── eedm/
            │   ├── __init__.py
            │   ├── __main__.py
            │   ├── AutoParallelizePy/
            │   │   ├── __init__.py
            │   │   ├── libs/
            │   │   │   ├── __init__.py
            │   │   │   ├── domainDecomposeND.py
            │   │   │   ├── funcs.py
            │   │   │   └── mpi4pyFuncs.py
            │   ├── io/
            │   │   ├── __init__.py
            │   │   ├── reader.py
            │   │   └── larexd/
            │   │       ├── __init__.py
            │   │       ├── read_lare_cfd2d.py
            │   │       └── read_lare_cfd3d.py
            │   ├── core/
            │   │   ├── __init__.py
            │   │   ├── EEDM_eq6.py
            │   │   ├── EEDM_eq9.py
            │   ├── methods/
            │   │   ├── __init__.py
            │   │   ├── differentiation.py
            │   ├── funcs/
            │   │   ├── __init__.py
            │   │   ├── eigenDecompFuncs.py
            │   │   ├── get_energy_sliced.py
            │   ├── prerequisites.py
            │   └── SI_constants.py
            └── settings.py
'''



import os, sys
import mpi4py

from . import system
from . import const
from . import context

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import settings

def main():
    if not settings.dimensional: const.mu0  = 1
    
    #◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
    #      Init Parallel                      #◈
    context.comm     = mpi4py.MPI.COMM_WORLD  #◈
    context.size     = context.comm.Get_size()#◈
    context.rank     = context.comm.Get_rank()#◈
    #◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
    
    datapath = system.add_slash(settings.datapath)
    outDirec = datapath + 'EEDM_results/'
    
    if context.rank == 0:
        err = system.bash("cd " + outDirec)
        if err != 0: system.bash("mkdir "+outDirec)
    
    
    if   settings.computationSwitch == "eq6": context.eq9Cond = False
    elif settings.computationSwitch == "eq9": context.eq6Cond = False
    else: pass
    
    if context.eq6Cond:
        from .core import EEDM_eq6
        EEDM_eq6.run(outDirec)
    
    context.comm.barrier()
    
    if context.eq9Cond:
        from .core import EEDM_eq9
        EEDM_eq9.run(outDirec)
    
    context.comm.barrier()
    
    if context.rank == context.mainrank: print("\nAll done!\n")


if __name__ == "__main__":
    main()