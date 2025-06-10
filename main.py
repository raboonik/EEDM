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
        │     ├── get_energy_sliced_parallel_funcs.py
        │     ├── prerequisites.py
        │     └── reader.py
        ├── main.py
        ├── settings.py
        └── SI_constants.py
    
    Input: snapshots.sdf
    
    Output1 (Eq6 of Paper III): 
        Processed HDF5 files containing the eigenenergy time derivatives (EEDM_<snapshot name>.h5):   
            {'Divx', 'Divy', 'Divz', 'Entx', 'Enty', 'Entz',
             'alf_x_1', 'alf_x_2', 'alf_y_1', 'alf_y_2', 'alf_z_1', 'alf_z_2',
             'slo_x_1', 'slo_x_2', 'slo_y_1', 'slo_y_2', 'slo_z_1', 'slo_z_2',
             'fas_x_1', 'fas_x_2', 'fas_y_1', 'fas_y_2', 'fas_z_1', 'fas_z_2'}
             
             • Here 1 means the - and 2 means the + characteristic directions
             
        Energy HDF5 files (Etot_<snapshot name>.h5):
            {'Kin', 'Mag', 'Int', 'Grv'}
            
            • The first energy file also stores all the grid information and metadata!
        
        Characteristic speed HDF5 files (Speed_<snapshot name>.h5):
            {'ax', 'ay', 'az', 'cfx', 'cfy', 'cfz', 'csx', 'csy', 'csz'}
        
    Output2 (Eq9 of Paper III; at specific 2D slabs):
        Processed 2D HDF5 files cut at specified points containing everything:
            
            Eigenenergies:
            {'Divx', 'Divy', 'Divz', 'Entx', 'Enty', 'Entz',
             'alf_x_1', 'alf_x_2', 'alf_y_1', 'alf_y_2', 'alf_z_1', 'alf_z_2',
             'slo_x_1', 'slo_x_2', 'slo_y_1', 'slo_y_2', 'slo_z_1', 'slo_z_2',
             'fas_x_1', 'fas_x_2', 'fas_y_1', 'fas_y_2', 'fas_z_1', 'fas_z_2'}
             
            Eigenenergy time derivatives:
            {'DDTDivx', 'DDTDivy', 'DDTDivz', 'DDTEntx', 'DDTEnty', 'DDTEntz',
             'DDTalf_x_1', 'DDTalf_x_2', 'DDTalf_y_1', 'DDTalf_y_2', 'DDTalf_z_1', 'DDTalf_z_2',
             'DDTslo_x_1', 'DDTslo_x_2', 'DDTslo_y_1', 'DDTslo_y_2', 'DDTslo_z_1', 'DDTslo_z_2',
             'DDTfas_x_1', 'DDTfas_x_2', 'DDTfas_y_1', 'DDTfas_y_2', 'DDTfas_z_1', 'DDTfas_z_2'}
             
            Characteristic speeds:
            {'ax', 'ay', 'az', 'cfx', 'cfy', 'cfz', 'csx', 'csy', 'csz'}
            
            Cell-center grid information:
            {'xc', 'yc', 'zc'}
            
            Time:
            {'time', 'timesteps'}
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