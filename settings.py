from libs import prerequisites as pre
'''
    Description: Settings for the parallelized EEDM code.
                 Currently only works for LareXd (>V3; sdf and cfd extensions).
                 Contact author to integrate your simulation code.
    
    Author: Abbas Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://arxiv.org/abs/2502.16010
    
    Assumptions:
        0) input data is the raw simulation output, do not need the user to pre-process anything. This adds some initial complexity but offers
           more portability and extension to other codes. We will have to define code-specific remapping onto cell-center grids for staggered codes
        1) the simulation code and the extension of the snapshots determine how all the data is read.
        2) the remapping
'''

simCode   = "lare"
dataExt   = "sdf"
# dataExt   = "cfd"
datapath  = "/home/abbas/temp/3DTorsionalAlfven_lare3d_3.4.1_Arbitrary_uniformIsoVertical_smooth_stratifiedVertical_new/Data/"
# datapath  = "/home/abbas/temp/Lucas_GT_subregion_stratifiedVertical"

# Data dimensionality (True: dimensional, False: non-dimensional)
dimensional = False

# Simulation dimensions
dim = 3

# Gravitational acceleration along -z in your code's units: MAKE SURE "g" IS INDEED A CONSTANT IN THE SIMULATION
g = 0
g = 274.
g = 274 * 1.5e5 * pre.mf * pre.mh / 5778.
g = 274 * 0.003792218351078803

gamma = 5/3


# For now only use the following mode
mode = "XYZUpDownSeparated"

# Select the set of equations to be computed
    # Options: "eq6", "eq9", "both"
    # eq9 depends on eq6
computationSwitch = "both"
# computationSwitch = "eq9"

# If eq9 to be computed, choose the slab planes and the slicing point locations across these plane-normal direction
# Since time integration needs to be performed, this slicing makes the computation faster. Simply select all the points
# along the plane-normal direction to run the computation for the entire 3D domain (MASSIVE DATA WARNING!)
slicingPlane = "xz"
slicingPnts  = [-1.5, 0., 1.1, 2.3] # Measured in code's length units
slicingPnts  = pre.np.arange(-0.175,0.175,0.05)
slicingPnts  = [0.]

# The first (initial state) and the last snapshots in the series to analyze: [0,-1] to include all snapshots
dataInterval = [0,-1]

# Number of snapshots to skip in between; 1 to not skip
skip = 1

# Cropping frame in the non-dimensional length unit: [0,0] for no cropping
cropFramex = [0,0]
cropFramey = [0,0]
cropFramez = [0,0]
# cropFramex = [-10,10]
# cropFramey = [-10,10]
# cropFramez = [80,90]


# Switch to compute the error term from paper II, unimportant for most purposes as long as DivB is sufficiently small
divBCond = False