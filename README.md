# Eigenenergy Decomposition Method (EEDM)
EEDM is an exact 3D method of breaking down the total energy density associated with compound nonlinear gravitational ideal-MHD disturbances into the energy contributions carried by each eigenmode. For more details, see [Paper 1](https://arxiv.org/abs/2502.16010), [Paper 2](https://iopscience.iop.org/article/10.3847/1538-4357/ad8dc8/meta) , and [Paper 3](https://iopscience.iop.org/article/10.3847/1538-4357/ad3bb6). 

# Author information
Axel Raboonik

Email: raboonik@gmail.com

Git:   https://github.com/raboonik

# Important note
This version currently only works with Lare3D V2 and V3, and supports the .sdf and .cfd snapshot extensions. Get in touch if you want me to add support for your code!

# How to use
## Step 0
Make sure the following python3 packages are installed

&nbsp;&nbsp;&nbsp;&nbsp; Numpy

&nbsp;&nbsp;&nbsp;&nbsp; SciPy 1.10.1 (pip install --force-reinstall -v "scipy==1.10.1")

&nbsp;&nbsp;&nbsp;&nbsp; mpi4py

## Step 1
git clone https://github.com/raboonik/EEDM.git   [DestinationDir]

## Step 2
Modify settings.py

cd [DestinationDir]

## Step 3
mpirun -n [#cores] python3 EEDM.py
