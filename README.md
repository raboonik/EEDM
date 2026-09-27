# Eigenenergy Decomposition Method (EEDM)
EEDM is an exact 3D method of breaking down the total energy density associated with compound nonlinear gravitational ideal-MHD disturbances into the energy contributions carried by each eigenmode. For more details, see [**Paper 1**](https://iopscience.iop.org/article/10.3847/1538-4357/ad3bb6), [**Paper 2**](https://iopscience.iop.org/article/10.3847/1538-4357/ad8dc8/meta) , and [**Paper 3**](https://iopscience.iop.org/article/10.3847/1538-4357/adc917). For the MPI-parallelized domain decomposition, the [AutoParallelizePy](https://github.com/raboonik/AutoParallelizePy) Python plug-in is used.

# Citation
Please use the following `.bib` entries to cite the papers

```text
@article{Raboonik_2024a,
doi = {10.3847/1538-4357/ad3bb6},
url = { https://doi.org/10.3847/1538-4357/ad3bb6 },
year = {2024a},
month = {may},
publisher = {The American Astronomical Society},
volume = {967},
number = {2},
pages = {80},
author = {Raboonik, Axel and Tarr, Lucas A. and Pontin, David I.},
title = {Exact Nonlinear Decomposition of Ideal-MHD Waves Using Eigenenergies},
journal = {The Astrophysical Journal}
}

@article{Raboonik_2024b,
doi = {10.3847/1538-4357/ad8dc8},
url = { https://doi.org/10.3847/1538-4357/ad8dc8 },
year = {2024b},
month = {dec},
publisher = {The American Astronomical Society},
volume = {977},
number = {2},
pages = {191},
author = {Raboonik, Axel and Pontin, David I. and Tarr, Lucas A.},
title = {Exact Nonlinear Decomposition of Ideal-MHD Waves Using Eigenenergies. II. Fully Analytical Equations and Pseudoadvective Eigenenergies},
journal = {The Astrophysical Journal}
}

@article{Raboonik_2025,
doi = {10.3847/1538-4357/adc917},
url = { https://doi.org/10.3847/1538-4357/adc917 },
year = {2025},
month = {may},
publisher = {The American Astronomical Society},
volume = {985},
number = {1},
pages = {102},
author = {Raboonik, Axel and Pontin, David I. and Tarr, Lucas A.},
title = {Exact Nonlinear Decomposition of Ideal-MHD Waves Using Eigenenergies. III. Gravity, Generalized Inhomogeneous Quasi-linear Partial Differential Equations, Mode Conversion, and Numerical Implementation},
journal = {The Astrophysical Journal}
}
```

# Author information
Axel Raboonik

Email: raboonik@gmail.com

GitHub:   https://github.com/raboonik

# Important notes

**1)** When cloning, use the **`--recurse-submodules`** flag (see Step 1 below).

**2)** This version works with 3D data from `Lare3D` V2, V3, and V4 (`.sdf` and `.cfd`), [`Mancha`](https://gitlab.com/Mancha3D/mancha) (`.h5`), and the [`Pencil Code`](https://github.com/pencil-code/pencil-code) (`VAR` snapshots in `data/allprocs`). 2D and 1D problems are analysed with the same 3D code, from 3D output with fewer points along the directions in which the problem does not vary (see [2D and 1D runs](#2d-and-1d-runs)). Get in touch to add support for your MHD simulation code!

# How to run
## Step 0: Required python3 libraries
Make sure the following `Python3` packages are installed

&nbsp;&nbsp;&nbsp;&nbsp; `numpy`

&nbsp;&nbsp;&nbsp;&nbsp; `h5py`

&nbsp;&nbsp;&nbsp;&nbsp; `SciPy`

&nbsp;&nbsp;&nbsp;&nbsp; `mpi4py`

&nbsp;&nbsp;&nbsp;&nbsp; **Note for LareXd > V3 users:** If you are working with `.sdf` files, it is assumed that a compatible Python interface for reading this format using `sdf.read` already exists in your environment. It must be built against the NumPy version you are using (a module built against NumPy 1.x will not import under NumPy 2.x).

&nbsp;&nbsp;&nbsp;&nbsp; **Note for Pencil Code users:** The Pencil Code's own Python package (`pencil-code/python`) must be on your `PYTHONPATH`.

## Step 1: Getting the code
&nbsp;&nbsp;&nbsp;&nbsp; `git clone --recurse-submodules https://github.com/raboonik/EEDM.git`

## Step 2: Setting up the analysis
&nbsp;&nbsp;&nbsp;&nbsp; `cd EEDM/`

&nbsp;&nbsp;&nbsp;&nbsp; Modify `settings.py` to set the analysis tasks (see [Settings](#settings) below)

## Step 3: Running the Code as a Package
Run the code from `EEDM/` using:

&nbsp;&nbsp;&nbsp;&nbsp; `mpirun -n [#cores] python3 -m eedm`

Each MPI process holds a full snapshot (and many derived arrays) in memory while computing Equation 6, so choose the number of processes according to the available memory rather than the number of cores.

At the start, EEDM prints a summary of the run (simulation code, data, number of snapshots, dimensions, the domain along x, y, and z (number of cells, length, and range of the cell centres, after cropping, in the units of the analysis), MPI processes, units, gravity, gamma, precision, and decomposition) and checks the settings, including the positions of the Equation 9 slabs, which it lists before Equation 6 starts. This way, mistakes are reported before any computation. An error on any MPI process stops the whole run.

### Note
EEDM always reads the `settings.py` in the directory it is run from, so run it from `EEDM/` as above, or from any directory holding its own copy of `settings.py`. If there is none, EEDM stops with an error saying so.

To run EEDM from directories other than `EEDM/` (for example, one directory with its own `settings.py` per analysis), install the package, in either regular or *editable* mode (editable means changes to the code take effect without reinstalling):

&nbsp;&nbsp;&nbsp;&nbsp; `cd EEDM/`

&nbsp;&nbsp;&nbsp;&nbsp; `pip install -e .`

and then, from the directory holding your `settings.py`, run either `mpirun -n [#cores] python3 -m eedm` or `mpirun -n [#cores] eedm`.

# Settings
All settings are in the `settings.py` that EEDM reads (`EEDM/settings.py`, or your own copy; see the note above). The most important ones are:

&nbsp;&nbsp;&nbsp;&nbsp;**•** `simCode`, `dataExt`, `datapath`: the simulation code (`lare`, `mancha`, `mancha_0`, or `pencil`), its snapshot format (`sdf`, `cfd`, `h5`, or `var`), and the directory containing the snapshots (a trailing `/` is optional; paths may contain spaces).

&nbsp;&nbsp;&nbsp;&nbsp;**•** `dimensionality_switch`: whether the data are in physical units (`True`) or non-dimensional (`False`). With `True`, the SI value of `mu0` is used and the data are taken to be in SI (MANCHA's `mks` output, or Pencil data converted to SI); with `False`, `mu0 = 1` is used (e.g. LARE's normalised output).

&nbsp;&nbsp;&nbsp;&nbsp;**•** `verticalAxis`: the simulation's vertical axis, pointing up (`"x"`, `"y"`, or `"z"`, default `"z"`); gravity acts along its negative direction. EEDM's equations take the vertical to be z, so for `"x"` or `"y"` the data are rotated internally (a cyclic permutation of the axes), while all settings and outputs stay in the simulation's own axes: e.g. with `verticalAxis = "y"`, the gravity mode is `eq6_m9_y`, and crop frames, slicing planes, and invariant axes are given in the simulation's axes as usual.

&nbsp;&nbsp;&nbsp;&nbsp;**•** `g`: the constant gravitational acceleration along -`verticalAxis`, as a positive number in the **same units as the data**. EEDM does not convert `g`. For example, for LARE output normalised by a length `L0` and speed `v0` (listed in the run's `lare3d.dat`), use `g = g_phys * L0 / v0**2`, e.g. `274 * L0 / v0**2` for solar surface gravity. Use `g = 0` for runs without gravity. A negative `g` stops the run with an error, since EEDM always takes gravity to point down, along -`verticalAxis`: give its magnitude, e.g. `g = 274` even if the simulation stores it as `-274` (like Pencil's `gravz`).

&nbsp;&nbsp;&nbsp;&nbsp;**•** `gamma`: the adiabatic index used for the sound speed and the internal energy `p / (gamma - 1)`.

&nbsp;&nbsp;&nbsp;&nbsp;**•** `invariantAxes`: the directions along which the simulation does not vary, for 2D and 1D runs (`[]` for 3D, e.g. `["y"]` for 2D in the x-z plane, `["y", "z"]` for 1D along x); see [2D and 1D runs](#2d-and-1d-runs).

&nbsp;&nbsp;&nbsp;&nbsp;**•** `precision`: `"float64"` (default, recommended) or `"float32"`. All data, computations, and outputs use this precision (except the time series `time` and `timesteps` and the crop indices `framec` and `frameb`, which are always float64). `"float32"` roughly halves memory and disk use but keeps only about 7 significant digits.

&nbsp;&nbsp;&nbsp;&nbsp;**•** `dataInterval`, `skip`: the first and last snapshot indices to analyse, both included (`[0, -1]` for all), and the stride between them.

&nbsp;&nbsp;&nbsp;&nbsp;**•** `cropFramex/y/z`: optional crop limits `[lower, upper]` in the units of the grid (`[0, 0]` for no cropping). Each limit snaps to the nearest grid point (for LARE, the nearest cell face), and both limits are included. The run header shows the resulting domain. Near any edge of the analysed domain, the spline derivatives only see one side, so the eigenpowers in the outermost few cells are unreliable: in the tests, the error falls by about 4 times per cell, from order 100% at the edge to about 1% three cells in and 0.1% five cells in. Crop a few cells wider than your region of interest, and disregard the outermost cells.

&nbsp;&nbsp;&nbsp;&nbsp;**•** `slicingPlane`, `slicingPntsType`, `slicingPnts`: the plane (`"xy"`, `"xz"`, or `"yz"`) of the 2D slabs on which Equation 9 is computed, and their positions along the plane's normal. With `slicingPntsType = "coordinate"`, `slicingPnts` are positions in the units of the grid, each snapped to the nearest grid point (`[0]` and `[0.0]` both mean the slab at 0). With `slicingPntsType = "index"`, they are grid indices (integers), counted from the start of the **cropped** grid. A position outside the (cropped) domain, an index outside the grid, or an empty list stops the run with an error, which appears at the start of the run, before Equation 6 is computed. Points that map to the same grid point give a single slab, with a note in the output. The slabs used are listed at the start of the run.

&nbsp;&nbsp;&nbsp;&nbsp;**•** `decomposition`: how the eigenenergies are grouped in the outputs (see the Equation 6 output keys below): `"MHD_modes_xyz"` (each mode separately along each direction x, y, z), `"MHD_branches_xyz"` (the reverse and forward modes of each branch added, along each direction), or `"MHD_net_branches"` (each branch summed over reverse and forward and over x, y, z). `"MHD_modes_xyz"` is the most informative, since the other two can always be obtained from it by summing its outputs.

&nbsp;&nbsp;&nbsp;&nbsp;**•** `computationSwitch`: `"eq6"`, `"eq9"`, or `"both"`.

&nbsp;&nbsp;&nbsp;&nbsp;**•** `savePredecompE`, `saveSpeeds`, `saveDivB`, `savePolytropK`, `saveDivBErr`: switches for the optional outputs described below.

&nbsp;&nbsp;&nbsp;&nbsp;**•** `waveFieldTolerance` (optional, default `1e-12`): the threshold of the no-wave-field mask (see "Points without a wave field" below); `0` switches the mask off.

&nbsp;&nbsp;&nbsp;&nbsp;**•** `verbose` (optional, default `False`): also print the estimated memory use of each MPI process and the MPI domain decomposition.

**Rerunning in the same directory:** Equation 9 uses only the Equation 6 files of the snapshots selected by the current `dataInterval` and `skip`, and ignores any others in the folder (with a note). When running Equation 9 on its own (`computationSwitch = "eq9"`), it uses the grid and cropping of the Equation 6 run that produced the files (stored in `grid_params.h5`), so the slab positions refer to that grid, and it needs the same `decomposition`; a different `decomposition` stops the run with an error.

# 2D and 1D runs
EEDM always solves the 3D equations. A 2D or 1D problem is analysed from 3D output (e.g. a 3D code run with only a few cells along the ignorable directions), with those directions listed in `invariantAxes`. Then:

&nbsp;&nbsp;&nbsp;&nbsp;**•** derivatives along the listed directions are zero, while all three components of the velocity and magnetic field are kept (2.5D MHD);

&nbsp;&nbsp;&nbsp;&nbsp;**•** each listed direction may have any number of grid points. EEDM checks that every field is uniform along it (to 1 part in 10^6 of the field's largest magnitude, otherwise it stops with an error) and keeps a single cell, so the outputs have length 1 along it;

&nbsp;&nbsp;&nbsp;&nbsp;**•** every direction not listed needs at least 4 grid points (for the spline derivatives), the vertical axis (`verticalAxis`) cannot be listed when `g > 0`, and the listed directions cannot be cropped (their `cropFrame` must be `[0, 0]`);

&nbsp;&nbsp;&nbsp;&nbsp;**•** the outputs along the listed directions are identically zero and stored as `[0]` placeholders (e.g. `eq6_m3_y` for `invariantAxes = ["y"]`), in the Equation 6 files and the Equation 9 slabs;

&nbsp;&nbsp;&nbsp;&nbsp;**•** for Equation 9, choose a `slicingPlane` that contains the varying direction(s), e.g. `"xz"` with `slicingPntsType = "index"` and `slicingPnts = [0]` for `invariantAxes = ["y"]`.

**Parallel propagation:** where the magnetic field is (nearly) along the direction q, the Alfven and slow modes along q (the Alfven and fast modes when the Alfven speed exceeds the sound speed) travel at the same speed and cannot be told apart. Their split then follows the direction of the tiny perpendicular field and is erratic, while their sum is well defined. Use the sum (e.g. `decomposition = "MHD_branches_xyz"` and `eq6_A_q + eq6_s_q`) in such cases, which are common in 1D problems.

# Outputs
As mentioned, the current version of the code handles `lareXd` (`sdf` and `cfd` snapshot extensions), `Mancha` (`h5` snapshot extension), and the `Pencil Code` (`VAR` snapshots), and outputs `h5` files containing the eigenpowers (eigenenergy time derivatives) and/or the eigenenergies themselves associated with the nine gravitational-ideal-MHD modes in the three x, y, and z Cartesian directions, either separately or grouped as set by `decomposition`. The computations are done according to Equations 6 and 9 of [**Paper 3**](https://iopscience.iop.org/article/10.3847/1538-4357/adc917). If g = 0, the code reduces to the non-gravitational ideal-MHD equations of [**Paper 2**](https://iopscience.iop.org/article/10.3847/1538-4357/ad8dc8). Note that although the EEDM method is valid for any 3D gravitational field (prescribed by a gravitational potential), the current version of the code is tailored to solar/stellar atmospheric use assuming uniform gravity along the negative vertical axis (`verticalAxis`, z by default).

All outputs are written to `<datapath>/EEDM_results/` (for the Pencil Code, `datapath` is the run's `data/allprocs/` directory). The main outputs are the eigenpowers (Equation 6), eigenenergies (Equation 9), and grid/parameter data. These are stored in the following three separate locations: 

&nbsp;&nbsp;&nbsp;&nbsp;**Equation 6:** `<datapath>/EEDM_results/Decomposed_EigenEnergies/PaperIII_Eq_6_3D/`

&nbsp;&nbsp;&nbsp;&nbsp;**Equation 9:** `<datapath>/EEDM_results/Decomposed_EigenEnergies/PaperIII_Eq_9_2DSlabs/`

&nbsp;&nbsp;&nbsp;&nbsp;**Grid/parameters:** `<datapath>/EEDM_results/grid_params.h5`

For 3D simulation data, the eigenpowers (Equation 6) are computed in 3D. However, due to the storage-intensive nature of the computations, the eigenenergies (Equation 9) are only computed on 2D slabs set by the user. The user is given the option to compute the eigenenergies (Equation 9) on one or more 2D slabs. This is controlled by the `slicingPlane`, `slicingPntsType`, and `slicingPnts` settings.

Note that computing Equation 6 is a prerequisite for computing Equation 9 (as the latter is just the time-integral of the former).

The file `grid_params.h5` contains the (cropped) cell-centred grid `xc`, `yc`, `zc` and crop indices `framec` (plus the staggered grid `xb`, `yb`, `zb` and `frameb` for LARE), and the attributes `g`, `gCond`, `gamma`, `decomposition`, `mu0`, `waveFieldTolerance`, `verticalAxis`, and `invariantAxes` used in the analysis.

## Other optional outputs 

The optional outputs below (of which the first two are more important) are switched on in `settings.py`. These are

&nbsp;&nbsp;&nbsp;&nbsp;**•** `savePredecompE`: the pre-decomposition terms of the total energy (i.e., the kinetic, internal, magnetic, and gravitational energies) stored in `<datapath>/EEDM_results/Predecomposition_Energies`. These are frequently required to evaluate the accuracy of the EEDM analysis (see [Paper 3](https://iopscience.iop.org/article/10.3847/1538-4357/adc917)). 

&nbsp;&nbsp;&nbsp;&nbsp;**•** `saveSpeeds`: the characteristic speeds `aq, csq, cfq` (for q in [x,y,z]) stored in `<datapath>/EEDM_results/Char_Speeds`

&nbsp;&nbsp;&nbsp;&nbsp;**•** `saveDivB` and `savePolytropK`: the magnetic field divergence (as measured by the numerical differentiation used in the code) and the polytropic coefficient `p / rho^gamma`, with their statistics, stored in `<datapath>/EEDM_results/Extras`

&nbsp;&nbsp;&nbsp;&nbsp;**•** `saveDivBErr`: the divB error term of [Paper 2](https://iopscience.iop.org/article/10.3847/1538-4357/ad8dc8), added to the Equation 6 and Equation 9 outputs as `eq6_m1_err` and `eq9_m1_err`

## Output keys of Equation 6 ("EigenenergyDDT_<|snapshot_name|>.h5")
`<|snapshot_name|>` is the snapshot file name without its extension (for the Pencil Code, `VAR<N>` with `N` zero-padded). The attribute `time` holds the snapshot time.

### Eigenpowers, by `decomposition` (set in `settings.py`; q in [x,y,z]):
```text
                "MHD_modes_xyz"   "MHD_branches_xyz"   "MHD_net_branches"
    Divergence: "eq6_m1_q"        "eq6_div_q"          "eq6_div"
    Entropy   : "eq6_m2_q"        "eq6_ent_q"          "eq6_ent"
    Alfven-   : "eq6_m3_q"        "eq6_A_q"            "eq6_A"
    Alfven+   : "eq6_m4_q"          (m3_q + m4_q)        (sum over - and +, and x, y, z)
    slow-     : "eq6_m5_q"        "eq6_s_q"            "eq6_s"
    slow+     : "eq6_m6_q"          (m5_q + m6_q)
    fast-     : "eq6_m7_q"        "eq6_f_q"            "eq6_f"
    fast+     : "eq6_m8_q"          (m7_q + m8_q)
    Gravity   : "eq6_m9_z"        "eq6_g_z"            "eq6_g"      (along the vertical: e.g. "eq6_m9_y" for verticalAxis = "y")
```
`"MHD_modes_xyz"` is the most informative: the other two decompositions can always be obtained from it by summing its outputs. With `"MHD_modes_xyz"`, the gravity mode along the two horizontal axes is also stored, as `[0]` (e.g. `eq6_m9_x` and `eq6_m9_y` for `verticalAxis = "z"`), since gravity acts along the vertical; with g = 0 the gravity output is stored as `[0]`.

### Other keys:
```text
    divB error term (if saveDivBErr): "eq6_m1_err"
    no-wave-field mask              : "maskWave"
```

Here `-` and `+` denote the reverse and forward modes, with eigenvalues `v_q - |a_q|`, `v_q - c_s,q`, `v_q - c_f,q` and `v_q + |a_q|`, `v_q + c_s,q`, `v_q + c_f,q`, respectively (Paper 3, Table 1).

**Points without a wave field:** where the flow is negligible, `|v| < waveFieldTolerance * sqrt(a^2 + c^2)` (a relative test, so it means the same in any units; `waveFieldTolerance` is an optional setting, default `1e-12`, `0` switches the mask off, and the value used is stored in `grid_params.h5`), the Alfven, slow, and fast eigenpowers (m3 to m8) are set to zero and the point is flagged in `maskWave`. Without this, a plasma at rest in a stratified atmosphere gives equal and opposite reverse/forward eigenpowers that cancel.

**Degenerate points:** where the eigenvectors of Paper 3 are not unique, the code uses the following conventions (after [Roe & Balsara 1996](https://doi.org/10.1137/S003613999427084X)): `beta = 1/sqrt(2)` where the perpendicular Alfven speed vanishes (Paper 3, Equation 7), `s_q = sgn(a_q) = +1` where `a_q = 0`, and `alpha_s = alpha_f = 1/sqrt(2)` at the triple point (`a_perp,q = 0` and `|a_q| = c`). These choices only decide how energy is split between modes that cannot be told apart at those points; the Alfven + slow + fast total, and the sum over all modes, do not depend on them.

## Output keys of Equation 9 ("EE_Slab\_<|slicingPlane|>\_n<|normal|>0\_<|index|>.h5")
The file name records the plane, its normal axis, and the grid index of the slab along that normal, e.g. `EE_Slab_xz_ny0_192.h5` for the xz-plane at y-index 192. These 2D-slabs contain both the eigenpowers and eigenenergies on the specified 2D planes, as functions of the two in-plane coordinates and time.

### Eigenenergies:
The same names as Equation 6 with the prefix `eq9_` for the chosen `decomposition`, e.g. `"eq9_m3_x"`, `"eq9_A_x"`, or `"eq9_A"`. The gravity output is left out when g = 0. Equation 9 must be run with the same `decomposition` as the Equation 6 files it reads, otherwise it stops with an error.

### Other keys:
```text
    Eigenpowers on the slab                  : the eq6_ outputs of the decomposition (same naming as Equation 6)
    divB error term (if saveDivBErr)         : "eq6_m1_err", "eq9_m1_err"
    Time and time steps                      : "time", "timesteps"
    Grid                                     : "xc", "yc", "zc"
    Total energy (if savePredecompE)         : "Etot"
    Characteristic speeds (if saveSpeeds)    : "ax", "ay", "az", "csx", "csy", "csz", "cfx", "cfy", "cfz", and the sound speed "cs"
    divB statistics vs time (if saveDivB)    : "maxDivB", "meanDivB", "stdDivB"
    Entropy statistics vs time (if savePolytropK): "maxEntropy", "meanEntropy", "stdEntropy"
```
The attributes record the slab (`slicingPlane`, `slicingPnt` = index, `slicingPntL` = coordinate) and the grid sizes.

## Output keys of the characteristic speeds ("Speed_<|snapshot_name|>.h5")
### q-directed speeds (for q in [x,y,z]): 
```text
    Alfven: "aq"
    slow  : "csq"
    fast  : "cfq"
```

## Output keys of kinetic, internal, magnetic, and gravitational energies ("Etot_<|snapshot_name|>.h5")
```text
    Kinetic : "Kin"
    Internal: "Int"
    Magnetic: "Mag"
    Gravity : "Grv"   (stored as [0] when g = 0)
```

## Output keys of the extras ("Extras_<|snapshot_name|>.h5")
```text
    divB (if saveDivB)                           : "divB"
    polytropic coefficient (if savePolytropK)    : "poly_k"
    attributes (if saveDivB)                     : "maxAbsDivB", "meanAbsDivB", "stdAbsDivB"
    attributes (if savePolytropK)                : "maxAbsEntropy", "meanAbsEntropy", "stdAbsEntropy"
```

# Important Note For Mancha Users

## Units
Mancha output is in SI (`mks`), so use `dimensionality_switch = True` and give `g` in m/s^2.

## Initial/background plasma (user input snapshot 0)
For all versions of Mancha, we assume that the user input snapshot prescribing the background plasma is stored in a single `h5` file named `"background_plasma.h5"` stored under the same directory as the main simulation snapshots. This file must contain all the initial values of `rho`, `vx`, `vy`, `vz`, `bx`, `by`, `bz`, and `pe` (using the same keys as standard Mancha snapshots), and is added to every snapshot. The sum is formed in the chosen `precision`, so snapshots and background stored in single precision (float32) keep the full `float64` accuracy of the sum. The user may flag the special case where all of these variables are zero, or where the snapshots already contain the full fields, by setting `simCode = mancha_0`. In this case, the inclusion of `"background_plasma.h5"` becomes redundant.

## Mancha versions
The latest version of Mancha outputs two separate `H5` snapshots at each data-dumping time step, namely, `<|snapshot_timestep|>_i.h5` and `<|snapshot_timestep|>_i_extra.h5`, where we assume that the former contains `rho`, `vx`, `vy`, `vz`, `bx`, `by`, `bz`, and the latter contains the gas pressure `pe`. If you are using this version, note that the EEDM will need the "extra" snapshots to read the gas pressure from. Each snapshot `X.h5` is paired with `X_extra.h5`; if any snapshot is missing its extra file, the code stops with an error.

# Important Note For LareXd Users

LARE output is normalised, so use `dimensionality_switch = False` (i.e. `mu0 = 1`) and give `g` in LARE's normalised units, `g = g_phys * L0 / v0**2`, where `L0` and `v0` are the normalising length and speed listed in the run's `lare3d.dat`. The staggered velocity and magnetic field are averaged onto the cell centres before the analysis.

# Important Note For Pencil Code Users

Set `simCode = "pencil"`, `dataExt = "var"`, and point `datapath` to the run's `data/allprocs/` directory, which holds the snapshots `VARN` and the run's `grid.dat`. Its parent `data/` directory must also contain the run's metadata files `dim.dat`, `param.nml`, `param2.nml`, and `index.pro`, which are needed to read the snapshots. The snapshots are read in time order (use `dataInterval` and `skip` as usual), with the ghost zones removed. The velocity and density (from `lnrho`) are read directly, the pressure is computed from the density and temperature (from `lnTT`) including partial hydrogen ionisation, and the magnetic field is computed as the curl of the vector potential plus the uniform external field `b_ext` (all three components). Runs that add any other external field (e.g. a precessing, moving-layer, global, dipole, or stratified field) stop with an error, since the reader cannot reproduce them.

## Units
* `dimensionality_switch = True`: everything is converted to SI using the run's `unit_*` parameters, so `g` (in m/s^2), the crop frames, and the slicing points must be given in SI.
* `dimensionality_switch = False`: the data stay in Pencil's code units, so `g`, the crop frames, and the slicing points must be given in code units. Since EEDM uses `mu0 = 1` for non-dimensional data while Pencil's code-unit `mu0` is the run's `mu0` parameter, the magnetic field is rescaled by `1/sqrt(mu0)` so that the magnetic energy and Alfven speed are unchanged.

If the run has constant gravity along the vertical axis (e.g. `gravz_profile = 'const'` for `verticalAxis = "z"`, or `gravy_profile` for `"y"`), EEDM warns when `g` does not match the run's gravity (e.g. `-gravz`) in the chosen units.

## Output file names
Pencil snapshots have no file extension, so the output files are labelled `VAR<N>` with `N` zero-padded (e.g. `EigenenergyDDT_VAR000500.h5`), which keeps them in time order.

## Equation of state
The internal energy is computed as `p / (gamma - 1)` using `gamma` from `settings.py`. If your run uses a non-ideal equation of state (e.g. with hydrogen ionization), this is an approximation.
