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
import numpy as np
from eedm import SI_constants as const

# MHD simulation code: options:
#   1. lare
#   2. mancha (use mancha_0 if the background state is zero)
simCode   = "lare"

# Data extension
dataExt   = "sdf" # sdf, cfd, or h5 (mancha only takes h5 files)

# Data directory
datapath  = "/path/to/simulation/data/"

# Data dimensionality (True: dimensional, False: non-dimensional)
dimensional = True

# Simulation dimensions
dim = 3

# Gravitational acceleration along -z in your code's units: MAKE SURE "g" IS INDEED A CONSTANT IN THE SIMULATION
# g = 274 * 1.5e5 * const.mf * const.mh / 5778. / const.kb
g = 274

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
# Or to include a wide range of points: slicingPnts  = np.arange(-0.175,0.175,0.05)

# The first (initial state) and the last snapshots in the series to analyze: [0,-1] to include all snapshots
dataInterval = [0,-1]

# Number of snapshots to skip in between; 1 to not skip
skip = 1

# Switch to save the characteristic speeds and predecomposed energies
saveSpeeds       = True
savePredecompE   = True

# Switch to save the DivB and the polytropic coefficient (k)
saveDivB      = False
savePolytropK = False

# Cropping frame in code's length unit: [0,0] for no cropping
cropFramex = [0,0]
cropFramey = [0,0]
cropFramez = [0,0]

# Switch to compute the error term from paper II, unimportant for most purposes as long as DivB is sufficiently small
saveDivBErr = False