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
            │   │   │   └── mpi.py
            │   ├── core/
            │   │   ├── __init__.py
            │   │   ├── EEDM_eq6.py
            │   │   ├── EEDM_eq9.py
            │   ├── io/
            │   │   ├── __init__.py
            │   │   ├── reader.py
            │   │   ├── extensionLoader.py
            │   │   └── larexd/
            │   │       ├── __init__.py
            │   │       ├── lare.py
            │   │       ├── read_lare_cfd2d.py
            │   │       └── read_lare_cfd3d.py
            │   ├── methods/
            │   │   ├── __init__.py
            │   │   ├── diff.py
            │   ├── utils/
            │   │   ├── __init__.py
            │   │   ├── crop.py
            │   │   ├── filter.py
            │   │   ├── get_energy_sliced.py
            │   ├── ct.py
            │   ├── system.py
            │   ├── decorators.py
            │   └── SI_constants.py
            └── settings.py
'''



import os, sys
from mpi4py import MPI

from . import system
from . import const
from . import context as ct
from . import decorators

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import settings

# Initialize parallelization
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#      Init Parallel             #◈
ct.comm     = MPI.COMM_WORLD     #◈
ct.size     = ct.comm.Get_size() #◈
ct.rank     = ct.comm.Get_rank() #◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈

@decorators.log_call
@decorators.timeit
def main():
    if not settings.dimensional: const.mu0  = 1
    
    datapath    = system.add_slash(settings.datapath)
    outDirecL1  = datapath   + 'EEDM_results/'
    outDirecSp  = outDirecL1 + 'Char_Speeds/'
    outDirecEn  = outDirecL1 + 'Predecomposition_Energies/'
    outDirecEx  = outDirecL1 + 'Extras/'
    
    outDirecL2  = outDirecL1 + 'Decomposed_EigenEnergies/'
    outDirecEq6 = outDirecL2 + 'PaperIII_Eq_6_3D/'
    outDirecEq9 = outDirecL2 + 'PaperIII_Eq_9_2DSlabs/'
    
    dirDict = {"parent": outDirecL1, "eq6": outDirecEq6, "eq9": outDirecEq9, "speed": outDirecSp, "energy": outDirecEn, "extra": outDirecEx}
    
    # Create the output directories
    if ct.rank == ct.mainrank:
        # Level 1
        system.create_dir(outDirecL1)
        system.create_dir(outDirecL2)
        
        # Level 2
        system.create_dir(outDirecEq6)
        system.create_dir(outDirecEq9)
        
        # Conditional levels
        if ct.SpCond             : system.create_dir(outDirecSp)
        if ct.EnCond             : system.create_dir(outDirecEn)
        if ct.DbCond or ct.PkCond: system.create_dir(outDirecEx)
    
    ct.comm.barrier()
    
    if   settings.computationSwitch == "eq6": ct.eq9Cond = False
    elif settings.computationSwitch == "eq9": ct.eq6Cond = False
    else: pass
    
    if ct.eq6Cond:
        from .core import EEDM_eq6
        EEDM_eq6.run(dirDict)
    
    ct.comm.barrier()
    
    if ct.eq9Cond:
        from .core import EEDM_eq9
        EEDM_eq9.run(dirDict)
    
    ct.comm.barrier()
    
    if ct.rank == ct.mainrank: print("\nAll done!\n")


if __name__ == "__main__":
    main()