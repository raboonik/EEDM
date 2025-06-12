# Eigenenergy Decomposition Method (EEDM)
EEDM is an exact 3D method of breaking down the total energy density associated with compound nonlinear gravitational ideal-MHD disturbances into the energy contributions carried by each eigenmode. For more details, see [Paper 1](https://iopscience.iop.org/article/10.3847/1538-4357/ad3bb6/meta), [Paper 2](https://iopscience.iop.org/article/10.3847/1538-4357/ad8dc8/meta) , and [Paper 3](https://iopscience.iop.org/article/10.3847/1538-4357/adc917). For the MPI-parallelized domain decomposition, the [AutoParallelizePy](https://github.com/raboonik/AutoParallelizePy) python plug-in is used.

# Author information
Axel Raboonik

Email: raboonik@gmail.com

Git:   https://github.com/raboonik

# Important notes

**1)** When cloning, use the **`--recurse-submodules`** option.

**2)** This version currently only works with Lare3D V2 and V3, and supports the `.sdf` and `.cfd` snapshot extensions. Get in touch if you want me to add support for your MHD simulation code!


# How to run
## Step 0: Required pyton3 libraries
Make sure the following python3 packages are installed

&nbsp;&nbsp;&nbsp;&nbsp; Numpy

&nbsp;&nbsp;&nbsp;&nbsp; h5py

&nbsp;&nbsp;&nbsp;&nbsp; SciPy 1.10.1 (`pip3 install --force-reinstall -v "scipy==1.10.1"`)

&nbsp;&nbsp;&nbsp;&nbsp; mpi4py

&nbsp;&nbsp;&nbsp;&nbsp; > **Note for LareXd > V3 users:** If you are working with .sdf files, it is assumed that a compatible Python interface for reading this format already exists in your environment.

## Step 1: Getting the code
&nbsp;&nbsp;&nbsp;&nbsp; `git clone --recurse-submodules https://github.com/raboonik/EEDM.git   [DestinationDir]`

## Step 2: Setting up the analysis
&nbsp;&nbsp;&nbsp;&nbsp; `cd [DestinationDir]`

&nbsp;&nbsp;&nbsp;&nbsp; Modify `settings.py` to set the analysis tasks

## Step 3: Running the code
&nbsp;&nbsp;&nbsp;&nbsp; `mpirun -n [#cores] python3 EEDM.py`

# Outputs
The curent version of the code outputs h5 files containing the eigenenergies (and/or their time derivatives) associated with each of the nine gravitational-ideal-MHD modes in each of the three x, y, and z directions separately. The computations are done according to the Equations 6 and 9 of [Paper 3](https://iopscience.iop.org/article/10.3847/1538-4357/adc917). If g = 0, the code reduces to the non-gravitational ideal-MHD equations of [Paper 2](https://iopscience.iop.org/article/10.3847/1538-4357/ad8dc8/meta). 

The outputs of the eigenenergy time derivatives (Equations 6) and the eigenenergies themselves (Equation 9) are stored in separate directories: 

&nbsp;&nbsp;&nbsp;&nbsp;Equation6 -> [SimulationDir/EEDM_results]

&nbsp;&nbsp;&nbsp;&nbsp;Equation9 -> [SimulationDir/EEDM_results/energySliced]

Due to the storage-intensive nature of the analysis, the user is given the option to compute the eigenenergies (Equation 9; stored in [SimulationDir/EEDM_results/energySliced]) on one or more 2D slabs. This is controlled by the "slicingPlane" and "slicingPnts" variables in settings.py.

Note that computing Equation 6 is a prerequisite for computing Equation 9.

## Output keys of Equations 6 ("EigenenergyDDT_[snapshot_name].h5" found in [SimulationDir/EEDM_results])

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

## Output keys of Equations 9 (found in [SimulationDir/EEDM_results/energySliced])
These 2D-slabs contain both the eigenenergy time derivatives and the eigenenergies on the specified 2D planes.

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

## Output keys of the characteristic speeds ("Speed_[snapshot_name].h5" found in [SimulationDir/EEDM_results])
###q-directed speeds (for q in [x,y,z]): 
```text
    Aflven: "aq"
    slow  : "csq"
    fast  : "cfq"
    sound : "c"
```

## Output keys of kinetic, internal, magnetic, and gravity energies ("Etot_[snapshot_name].h5" found in [SimulationDir/EEDM_results])
```text
    Kinetic : "Kin"
    Internal: "Int"
    Magnetic: "Mag"
    Gravity : "Grv"
```

In addition to the above, the first Etot h5 datafile also contains the grid information, which varies from code for code. However, since the current EEDM code handles the computations by mapping the plasma state vector onto a regular cell-centered grid, it suffices to only read this data:
```text
    Cell-centered x-axis: "xc"
    Cell-centered y-axis: "yc"
    Cell-centered z-axis: "zc"
```