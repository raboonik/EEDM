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
        ├── libs/
        │     ├── AutoParallelizePy/
        │            ├── domainDecomposeND.py
        │            ├── funcs.py
        │            ├── mpi4pyFuncs.py
        │            └── prerequisites.py
        │     ├── larexd/
        │            ├── read_lare_cfd2d.py
        │            └── read_lare_cfd3d.py
        │     ├── differentiation.py
        │     ├── EEDM_eq6.py
        │     ├── EEDM_eq9.py
        │     ├── eigenDecompFuncs.py
        │     ├── get_energy_sliced.py
        │     ├── prerequisites.py
        │     ├── reader.py
        │     └── SI_constants.py
        ├── EEDM.py
        └── settings.py
'''

#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
import os, sys                                        #◈
sys.path.insert(0, os.path.join(os.getcwd(), 'libs')) #◈
from libs import prerequisites as pre                 #◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈

if not pre.dimensional: pre.mu0  = 1

smallestAlfvSpeedAllowed = 1.e-30
smallestWaveFieldAllowed = 1.e-30
epsilon = 9.e-15

#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#      Init Parallel                  #◈
comm     = pre.mpi4py.MPI.COMM_WORLD  #◈
size     = comm.Get_size()            #◈
rank     = comm.Get_rank()            #◈
mainrank = 0                          #◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈

datapath = pre.add_slash(pre.datapath)
outDirec = datapath + 'EEDM/'

if rank == 0:
    err = pre.bash("cd " + outDirec)
    if err != 0: pre.bash("mkdir "+outDirec)

eq6Cond = True
eq9Cond = True
if   pre.computationSwitch == "eq6": eq9Cond = False
elif pre.computationSwitch == "eq9": eq6Cond = False
else: pass

if eq6Cond:
    with open(os.getcwd()+"/libs/EEDM_eq6.py") as f:
        code = f.read()
        exec(code)


comm.barrier()

if eq9Cond:
    with open(os.getcwd()+"/libs/EEDM_eq9.py") as f:
        code = f.read()
        exec(code)

comm.barrier()

if rank == mainrank: print("\nAll done!\n")