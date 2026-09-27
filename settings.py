'''
    Description: Settings for the parallelized EEDM code.
                 Works with 3D data from LARE (V2 to V4; .sdf and .cfd), MANCHA (.h5), and the Pencil Code (VAR snapshots).
                 Contact the author to integrate your simulation code.
    
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
import numpy as np   # available for settings written as expressions, e.g. slicingPnts = np.arange(...)

# MHD simulation code: options:
#   1. lare   (needs the sdf library already compiled for your numpy version and available on your PYTHONPATH)
#   2. mancha (use mancha_0 if the background state is zero)
#   3. pencil (needs pencil-code/python on your PYTHONPATH)
simCode   = "lare"

# Data extension
dataExt   = "sdf" # sdf, cfd, or h5 (mancha only takes h5 files), var (pencil only takes var)

# Data directory
#   pencil: the run's data/allprocs/ directory (snapshots VARN are read from there).
#           Its parent data/ directory must also hold the run's dim.dat, param.nml, param2.nml and index.pro.
#           dimensionality_switch = True converts to SI (set g in m/s^2, crop frames and slicing points in m);
#           dimensionality_switch = False keeps code units (g, crop frames and slicing points in code units).
datapath  = "/path/to/simulation/data/"

# Data dimensionality (True: dimensional, False: non-dimensional)
dimensionality_switch = False

# Floating-point precision of the data, all computations, and the outputs: "float64" (recommended) or "float32"
# (float32 halves memory and disk use but keeps only ~7 significant digits)
precision = "float64"

# 2D and 1D runs: EEDM always solves the 3D equations. For a simulation that does not vary along one or two directions (e.g.
# a 3D code run with only a few cells along them), list those directions here. Derivatives along them are then zero, while all
# three components of v and B are kept. Each listed axis may have any number of grid points; EEDM checks that the data are
# uniform along it and keeps a single cell (its outputs along these directions are stored as [0] placeholders).
# Every direction not listed needs at least 4 grid points. The vertical axis cannot be listed when g > 0.
    # []         : 3D
    # ["y"]      : 2D in the x-z plane (similarly ["x"] or ["z"])
    # ["y", "z"] : 1D along x
invariantAxes = []

# The simulation's vertical axis, pointing up: "x", "y", or "z". Gravity acts along its negative direction. EEDM's equations
# take the vertical to be z, so for "x" or "y" the data are rotated internally, while all settings and outputs stay in the
# simulation's own axes (e.g. the gravity mode is then eq6_m9_y for "y")
verticalAxis = "z"

# Magnitude of the uniform gravitational acceleration, which EEDM takes to point down, along -verticalAxis (potential phi = g * height),
# in units compatible with the "dimensionality_switch". Must be >= 0: e.g. g = 274 for solar gravity pointing down, even if the
# simulation stores it as -274. Use g = 0 for no gravity.
# MAKE SURE "g" IS INDEED A CONSTANT IN THE SIMULATION
g = 0

# The adiabatic polytropic index
gamma = 5/3

# How the eigenenergies (Table 1 of Paper III) are decomposed in the outputs. Every branch has a reverse (-) and a
# forward (+) mode: Alfven m3 (-) and m4 (+), slow m5 (-) and m6 (+), fast m7 (-) and m8 (+); div (m1), ent (m2), and
# gravity (m9, along z-axis only in the code; full 3D description in the paper) are single modes
    # "MHD_modes_xyz"    : each mode separately along each direction x, y, z (m1_x, ..., m8_z, m9_z)
    # "MHD_branches_xyz" : reverse + forward of each branch, still along each direction (div_x, ent_x, A_x, s_x, f_x, ..., g_z)
    # "MHD_net_branches" : each branch summed over reverse + forward and over x, y, z (div, ent, A, s, f, g)
# "MHD_modes_xyz" is the most informative: the other two can always be recovered from it by summing its outputs (see Paper III).
decomposition = "MHD_modes_xyz"

# Select the set of equations to be computed
    # Options: "eq6", "eq9", "both"
    # eq9 depends on eq6
computationSwitch = "both"

# If eq9 to be computed, choose the slab planes and the slicing point locations across these plane-normal direction
# Since time integration needs to be performed, this slicing makes the computation faster. Simply select all the points
# along the plane-normal direction to run the computation for the entire 3D domain (MASSIVE DATA WARNING!)
slicingPlane = "xz"
# How slicingPnts is read:
    # "coordinate": positions along the plane-normal axis in the data's length unit (see "dimensionality_switch"), each snapped
    #               to the nearest grid point; e.g. [0] and [0.0] both mean the slab at y = 0
    # "index"     : grid indices along the plane-normal axis, counted from the start of the CROPPED grid (see cropFrame*)
# Points outside the (cropped) domain stop the run with an error; points that snap to the same grid point give one slab
slicingPntsType = "coordinate"
slicingPnts     = [0., 1.]
# Or to include a wide range of points: slicingPnts  = np.arange(-0.175,0.175,0.05)

# The first (initial state) and the last snapshots in the series to analyze: [0,-1] to include all snapshots
dataInterval = [0,-1]

# Number of snapshots to skip in between; 1 to not skip
skip = 1

# Switch to save the characteristic speeds and predecomposed energies
saveSpeeds     = True
savePredecompE = True

# Switch to save the DivB and the polytropic coefficient (k)
saveDivB      = False
savePolytropK = False

# Cropping frame in the data's length unit (see "dimensionality_switch"): [0,0] for no cropping
cropFramex = [0,0]
cropFramey = [0,0]
cropFramez = [0,0]

# Switch to compute the error term from paper II, unimportant for most purposes as long as DivB is sufficiently small
saveDivBErr = False

# Quiet-point mask: where |v| < waveFieldTolerance * sqrt(a^2 + c^2) (relative to the local fast-speed scale, so the same in any
# units), the point counts as having no wave field and its Alfven, slow, and fast eigenpowers are set to 0 (flagged in maskWave).
# Without it, a plasma at rest gives equal and opposite reverse/forward eigenpowers. Use a number >= 0; 0 switches the mask off.
waveFieldTolerance = 1e-12

# Print extra diagnostics while running: the estimated memory use of each rank and the MPI domain decomposition
verbose = False
