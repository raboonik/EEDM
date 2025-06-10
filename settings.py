from libs import prerequisites as pre
'''
    Description: Settings for the parallelized EEDM code.
                 Currently only works with Lare3d (V2 and V3; sdf and cfd extensions).
                 Contact author to integrate your simulation code.
    
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Main EEDM article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Assumptions:
        0) input data is the raw simulation output, do not need the user to pre-process anything. This adds some initial complexity but offers
           more portability and extension to other codes. We will have to define code-specific remapping onto cell-center grids for staggered codes
        1) the simulation code and the extension of the snapshots determine how all the data is read.
        2) the remapping
'''

simCode   = "lare"
dataExt   = "sdf" # sdf or cfd
datapath  = "/home/user/simulationDir/"
datapath  = "/home/abbas/temp/Lucas_GT_subregion_stratifiedVertical"

# Data dimensionality (True: dimensional, False: non-dimensional)
dimensional = False

# Simulation dimensions
dim = 3

# Gravitational acceleration along -z in your code's units: MAKE SURE "g" IS INDEED A CONSTANT IN THE SIMULATION
g = 274 * 1.5e5 * pre.mf * pre.mh / 5778.

gamma = 5/3

# For now only use the following mode
mode = "XYZUpDownSeparated"

# Select the set of equations to be computed
    # Options: "eq6", "eq9", "both"
    # eq9 depends on eq6
computationSwitch = "both"

# If eq9 to be computed, choose the slab planes and the slicing point locations across these plane-normal direction
# Since time integration needs to be performed, this slicing makes the computation faster. Simply select all the points
# along the plane-normal direction to run the computation for the entire 3D domain (MASSIVE DATA WARNING!)
slicingPlane = "xz"
slicingPnts  = [-1.5, 0., 1.1, 2.3] # Measured in code's length unit
# Or to include a wide range of points: slicingPnts  = pre.np.arange(-0.175,0.175,0.05)

# The first (initial state) and the last snapshots in the series to analyze: [0,-1] to include all snapshots
dataInterval = [0,-1]

# Number of snapshots to skip in between; 1 to not skip
skip = 1

# Cropping frame in code's length unit: [0,0] for no cropping
cropFramex = [0,0]
cropFramey = [0,0]
cropFramez = [0,0]

# Switch to compute the error term from paper II, unimportant for most purposes as long as DivB is sufficiently small
divBCond = False