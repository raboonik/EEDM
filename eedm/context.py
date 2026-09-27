'''
    Shared memory context
'''

import numpy as np

import settings as set

comm     = None
rank     = None
size     = None
mainrank = 0


eq6Cond  = True
eq9Cond  = True

# Auxiliary booleans for neater syntaxes
EnCond   = set.savePredecompE
SpCond   = set.saveSpeeds
ErCond   = set.saveDivBErr   # divB error term of Paper II (eq6_m1_err / eq9_m1_err)
DbCond   = set.saveDivB      # divB field and its statistics (Extras files)
PkCond   = set.savePolytropK

# Print the per-rank memory estimates and the MPI domain decomposition (optional in settings.py)
verbose  = bool(getattr(set, "verbose", False))

# Quiet-point mask: a point counts as having no wave field when |v| < waveFieldTolerance * sqrt(a^2 + c^2), i.e. relative
# to the local fast-speed scale, so the test means the same in any units (optional in settings.py; 0 switches the mask off)
waveFieldTolerance = getattr(set, "waveFieldTolerance", 1.e-12)
if isinstance(waveFieldTolerance, bool) or not isinstance(waveFieldTolerance, (int, float, np.integer, np.floating)) or not waveFieldTolerance >= 0:
    raise ValueError("settings.waveFieldTolerance = " + repr(waveFieldTolerance) + " is not supported. Use a number >= 0 " +
                     "(default 1e-12; 0 switches the quiet-point mask off).")
waveFieldTolerance = float(waveFieldTolerance)

# Floating-point precision used for the data, all computations, and the outputs
if set.precision not in ("float32", "float64"):
    raise ValueError('settings.precision = ' + repr(set.precision) + ' is not supported. Use "float64" (recommended) or "float32".')
dtype    = np.dtype(set.precision)

# How the eigenenergies are grouped in the outputs (see settings.py and utils.decomposition)
decompositions = ("MHD_modes_xyz", "MHD_branches_xyz", "MHD_net_branches")
_options       = ", ".join('"' + d + '"' for d in decompositions)
if not hasattr(set, "decomposition"):
    raise ValueError('settings.py has no "decomposition" setting (it replaces the former "mode"). Set it to one of ' + _options + '.')
if set.decomposition not in decompositions:
    _former = {"XYZUpDownSeparated": "MHD_modes_xyz", "UpDownCombined": "MHD_branches_xyz"}
    raise ValueError('settings.decomposition = ' + repr(set.decomposition) + ' is not supported. Use one of ' + _options + '.' +
                     (' The former "' + set.decomposition + '" is now "' + _former[set.decomposition] + '".' if set.decomposition in _former else ''))
decomposition = set.decomposition

# The simulation's vertical axis (pointing up; gravity acts along its negative direction). EEDM's equations take it to be z:
# for "x" or "y" the data are rotated into the EEDM frame when read and the outputs rotated back (see utils.orientation)
verticalAxis = getattr(set, "verticalAxis", "z")
if verticalAxis not in ("x", "y", "z"):
    raise ValueError("settings.verticalAxis = " + repr(verticalAxis) + ' is not supported. Use "x", "y", or "z": the axis pointing up, with ' +
                     "gravity acting along its negative direction.")

# The axis permutation from the simulation frame to the EEDM frame: EEDM axis k is simulation axis PERMUTATIONS[verticalAxis][k]
# (a cyclic permutation, i.e. a proper rotation; used by utils.orientation)
PERMUTATIONS = {"z": (0, 1, 2), "y": (2, 0, 1), "x": (1, 2, 0)}

# EEDM assumes a uniform gravitational acceleration of magnitude g pointing down, along -verticalAxis (potential phi = g * height).
# A negative g would silently describe gravity pointing up, so it is refused rather than guessed at.
if not float(set.g) >= 0:   # also catches nan
    raise ValueError("settings.g = " + str(set.g) + " is negative or not a number. settings.g is the magnitude of the uniform gravitational acceleration, "
                     "which EEDM takes to point down, along -" + verticalAxis + " (settings.verticalAxis), so it must be >= 0 (g = 0 for no gravity). "
                     "For example, a simulation with gravity of 274 m/s^2 pointing down needs g = 274, even if the code itself "
                     "stores it as -274 (e.g. Pencil's gravz). Gravity pointing up (along +" + verticalAxis + ") is not supported.")

# 2D and 1D problems run through the same 3D equations: the directions along which the simulation does not vary are listed
# in settings.invariantAxes (optional, default [] = fully 3D). Derivatives along them are zero, the fields keep all three
# components, and the reader reduces these axes to a single cell after checking that the data are indeed uniform along them.
invariantAxes = getattr(set, "invariantAxes", [])
_axes         = [] if isinstance(invariantAxes, str) else list(invariantAxes)
if isinstance(invariantAxes, str) or not all(a in ("x", "y", "z") for a in _axes) or len(_axes) != len(dict.fromkeys(_axes)) or len(_axes) > 2:
    raise ValueError("settings.invariantAxes = " + repr(invariantAxes) + ' is not supported. List up to two different axes from "x", "y", "z", ' +
                     'e.g. ["y"] for a 2D run in the x-z plane, ["y", "z"] for a 1D run along x, or [] for a 3D run.')
invariantAxes = tuple(_axes)
if hasattr(set, "dim"):
    if set.dim != 3 - len(invariantAxes):
        raise ValueError("settings.dim = " + repr(set.dim) + " is no longer used: the number of dimensions follows from settings.invariantAxes " +
                         '(e.g. ["y"] for a 2D run in the x-z plane). Remove dim from settings.py and set invariantAxes instead.')
if verticalAxis in invariantAxes and float(set.g) > 0:
    raise ValueError('settings.invariantAxes contains the vertical axis "' + verticalAxis + '" (settings.verticalAxis), but settings.g = ' + str(set.g) +
                     " > 0: gravity acts along the vertical, and a stratified atmosphere cannot be uniform along it. Use g = 0, or keep " +
                     verticalAxis + " as a varying direction.")
for _a in invariantAxes:
    if np.sum(np.abs(getattr(set, "cropFrame" + _a))) > 0:
        raise ValueError("settings.cropFrame" + _a + " = " + str(getattr(set, "cropFrame" + _a)) + ", but " + _a + " is listed in invariantAxes, " +
                         "so there is nothing to crop along it. Use cropFrame" + _a + " = [0, 0].")
dimensions = 3 - len(invariantAxes)

# The invariant axes in the EEDM frame (the frame of the equations; the same as the simulation frame when verticalAxis = "z")
invariantAxesE = tuple("xyz"[PERMUTATIONS[verticalAxis].index("xyz".index(a))] for a in invariantAxes)
