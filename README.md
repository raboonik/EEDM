# Eigenenergy Decomposition Method (EEDM)
EEDM is an exact 3D method of breaking down the total energy density associated with compound nonlinear gravitational ideal-MHD disturbances into the energy contributions carried by each eigenmode. For more details, see [**Paper 1**](https://iopscience.iop.org/article/10.3847/1538-4357/ad3bb6), [**Paper 2**](https://iopscience.iop.org/article/10.3847/1538-4357/ad8dc8/meta) , and [**Paper 3**](https://iopscience.iop.org/article/10.3847/1538-4357/adc917). For the MPI-parallelized domain decomposition, the [AutoParallelizePy](https://github.com/raboonik/AutoParallelizePy) Python plug-in is used.

# Citation
Please use the following `.bib` entries to cite the papers

@article{Raboonik_2024,
doi = {10.3847/1538-4357/ad3bb6},
url = {https://doi.org/10.3847/1538-4357/ad3bb6},
year = {2024},
month = {may},
publisher = {The American Astronomical Society},
volume = {967},
number = {2},
pages = {80},
author = {Raboonik, Axel and Tarr, Lucas A. and Pontin, David I.},
title = {Exact Nonlinear Decomposition of Ideal-MHD Waves Using Eigenenergies},
journal = {The Astrophysical Journal}
}

@article{Raboonik_2024,
doi = {10.3847/1538-4357/ad8dc8},
url = {https://doi.org/10.3847/1538-4357/ad8dc8},
year = {2024},
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
url = {https://doi.org/10.3847/1538-4357/adc917},
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






# Author information
Axel Raboonik

Email: raboonik@gmail.com

GitHub:   https://github.com/raboonik

# Important notes

**1)** When cloning, use the **`--recurse-submodules`** flag (see Step 1 below).

**2)** This version currently only works with `Lare3D` V2 and V3 (`.sdf` and `.cfd`), and [`Mancha`](https://gitlab.com/Mancha3D/mancha) (`.h5`). Get in touch to add support for your MHD simulation code!

# How to run
## Step 0: Required pyton3 libraries
Make sure the following `Python3` packages are installed

&nbsp;&nbsp;&nbsp;&nbsp; `numpy`

&nbsp;&nbsp;&nbsp;&nbsp; `h5py`

&nbsp;&nbsp;&nbsp;&nbsp; `SciPy`

&nbsp;&nbsp;&nbsp;&nbsp; `mpi4py`

&nbsp;&nbsp;&nbsp;&nbsp; **Note for LareXd > V3 users:** If you are working with `.sdf` files, it is assumed that a compatible Python interface for reading this format using `sdf.read` already exists in your environment.

## Step 1: Getting the code
&nbsp;&nbsp;&nbsp;&nbsp; `git clone --recurse-submodules https://github.com/raboonik/EEDM.git`

## Step 2: Setting up the analysis
&nbsp;&nbsp;&nbsp;&nbsp; `cd EEDM/`

&nbsp;&nbsp;&nbsp;&nbsp; Modify `settings.py` to set the analysis tasks

## Step 3: Running the Code as a Package
Run the code from `EEDM/` using:

&nbsp;&nbsp;&nbsp;&nbsp; `mpirun -n [#cores] python3 -m eedm`

### Note
You can also install the EEDM package and run it from anywhere (inside or outside of `EEDM/`) using the same command. To install in *editable* mode, run:

&nbsp;&nbsp;&nbsp;&nbsp; `cd EEDM/`

&nbsp;&nbsp;&nbsp;&nbsp; `pip install -e .`

# Outputs
As mentioned, the curent version of the code handles `lareXd` (`sdf` and `cfd` snapshot extensions) and `Mancha` (`h5` snapshot extension), and outputs `h5` files containing the eigenpowers (eigenenergy time derivatives) and/or the eigenenergies themselves associated with each of the nine gravitational-ideal-MHD modes in each of the three x, y, and z Cartesian directions separately. The computations are done according to Equations 6 and 9 of [**Paper 3**](https://iopscience.iop.org/article/10.3847/1538-4357/adc917). If g = 0, the code reduces to the non-gravitational ideal-MHD equations of [**Paper 2**](https://iopscience.iop.org/article/10.3847/1538-4357/ad8dc8). Note that although the EEDM method is valid for any 3D gravitational field (prescribed by a gravitational potential), the current version of the code is tailored to solar/stellar atmospheric use assuming uniform gravity along -z.

The main outputs are the eigenpowers (Equation 6), eigenenergies (Equation 9), and grid/parameter data. These are stored in the following three separate directories: 

&nbsp;&nbsp;&nbsp;&nbsp;**Equation 6:** `<|SimulationDir|>/EEDM_results/Decomposed_EigenEnergies/PaperIII_Eq_6_3D/`

&nbsp;&nbsp;&nbsp;&nbsp;**Equation 9:** `<|SimulationDir|>/EEDM_results/Decomposed_EigenEnergies/PaperIII_Eq_9_2DSlabs/`

&nbsp;&nbsp;&nbsp;&nbsp;**Grid/parameters:** `<|SimulationDir|>/EEDM_results/grid_params.h5`

For 3D simulation data, the eigenpowers (Equation 6) are computed in 3D. However, due to the storage-intensive nature of the computations, the eigenenergies (Equation 9) are only computed on 2D slabs set by the user. The user is given the option to compute the eigenenergies (Equation 9) on one or more 2D slabs. This is controlled by the "slicingPlane" and "slicingPnts" variables in `EEDM/settings.py`.

Note that computing Equation 6 is a prerequisite for computing Equation 9 (as the latter is just the time-integral of the former).

An additional file named `grid_params.h5` is outputted, which contains the meshgrid data and important parameters used in the analysis.

## Other optional outputs 

The optional outputs below (of which the first two are more important) can be switched on in the code. These are

&nbsp;&nbsp;&nbsp;&nbsp;**•** the pre-decomposition terms of the total energy (i.e., the kinetic, internal, magnetic, and gravitational energies) stored in `<|SimulationDir|>/EEDM_results/Predecomposition_Energies`. These are frequently required to evaluate the accuracy of the EEDM analysis (see [Paper 3](https://iopscience.iop.org/article/10.3847/1538-4357/adc917)). 

&nbsp;&nbsp;&nbsp;&nbsp;**•** the characteristic speeds `aq, csq, cfq` (for q in [x,y,z]) stored in `<|SimulationDir|>/EEDM_results/Char_Speeds`

&nbsp;&nbsp;&nbsp;&nbsp;**•** extra information on the polytropic coefficient and the magnetic field divergence as measured by the numerical differential method used in the code, stored in `SimulationDir/EEDM_results/Extras`

## Output keys of Equation 6 ("EigenenergyDDT_<|snapshot_name|>.h5")

###q-directed (for q in [x,y,z]): 
```text
    Divergence: "eq6_m1_q"
    Entropy   : "eq6_m2_q"
    Aflven-   : "eq6_m3_q"
    Aflven+   : "eq6_m4_q"
    slow-     : "eq6_m5_q"
    slow+     : "eq6_m6_q"
    fast-     : "eq6_m7_q"
    fast+     : "eq6_m8_q"
    Gravity   : "eq6_m9_z"
```

## Output keys of Equations 9 ("EE_Slab\_<|slicingPlane|>\_<|slicingPnts|>.h5")
These 2D-slabs contain both the eigenpowers and eigenenergies on the specified 2D planes.

###q-directed eigenenergies (for q in [x,y,z]): 
```text
    Divergence: "eq9_m1_q"
    Entropy   : "eq9_m2_q"
    Aflven-   : "eq9_m3_q"
    Aflven+   : "eq9_m4_q"
    slow-     : "eq9_m5_q"
    slow+     : "eq9_m6_q"
    fast-     : "eq9_m7_q"
    fast+     : "eq9_m8_q"
    Gravity   : "eq9_m9_z"
```

## Output keys of the characteristic speeds ("Speed_<|snapshot_name|>.h5")
###q-directed speeds (for q in [x,y,z]): 
```text
    Aflven: "aq"
    slow  : "csq"
    fast  : "cfq"
```

## Output keys of kinetic, internal, magnetic, and gravitational energies ("Etot_<|snapshot_name|>.h5")
```text
    Kinetic : "Kin"
    Internal: "Int"
    Magnetic: "Mag"
    Gravity : "Grv"
```

# Important Note For Mancha Users

## Initial/background plasma (user input snapshot 0)
For all versions of Mancha, we assume that the user input snapshot prescribing the background plasma is stored in a single `h5` file named `"background_plasma.h5"` stored under the same directory as the main simulation snapshots. This file must contain all the initial values of `rho`, `vx`, `vy`, `vz`, `bx`, `by`, `bz`, and `pe` (using the same keys as standard Mancha snapshots). The user may flag the special case where all of these variables are zero by setting `simCode = mancha_0`. In this case, the inclusion of `"background_plasma.h5"` becomes redundant.

## Mancha versions
The latest version of Mancha outputs two separate `H5` snapshots at each data-dumping time step, namely, `<|snapshot_timestep|>_i.h5` and `<|snapshot_timestep|>_i_extra.h5`, where we assume that the former contains `rho`, `vx`, `vy`, `vz`, `bx`, `by`, `bz`, and the latter contains the gas pressure `pe`. If you are using this version, note that the EEDM will need the "extra" snapshots to read the gas pressure from.
