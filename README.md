# Eigenenergy Decomposition Method (EEDM)
EEDM is an exact 3D method of breaking down the total energy density associated with compound nonlinear gravitational ideal-MHD disturbances into the energy contributions carried by each eigenmode. For more details, see [Paper 1](https://arxiv.org/abs/2502.16010), [Paper 2](https://iopscience.iop.org/article/10.3847/1538-4357/ad8dc8/meta) , and [Paper 3](https://iopscience.iop.org/article/10.3847/1538-4357/ad3bb6). 

# Author information
Axel Raboonik

Email: raboonik@gmail.com

Git:   https://github.com/raboonik

# Installation
## Step 0
Make sure the following python3 packages are installed

&nbsp;&nbsp;&nbsp;&nbsp; Numpy

&nbsp;&nbsp;&nbsp;&nbsp; SciPy 1.10.1 (pip install --force-reinstall -v "scipy==1.10.1")

&nbsp;&nbsp;&nbsp;&nbsp; mpi4py

## Step 1
1) Make the installation script (install.sh) executable by running

&nbsp;&nbsp;&nbsp;&nbsp;chmod +x install.sh

2) To install under a specific directory run (do not run as root)

&nbsp;&nbsp;&nbsp;&nbsp;./install.sh path/to/directory

&nbsp;&nbsp;&nbsp;&nbsp;or to install under the default directory simply execute

&nbsp;&nbsp;&nbsp;&nbsp;./install.sh

## Step 2
Run "add2path.sh" to automatically update the python enviornment by executing

&nbsp;&nbsp;&nbsp;&nbsp;. ./add2path.sh


# Uninstallation
To uninstall and update the python environment simply run

&nbsp;&nbsp;&nbsp;&nbsp;. ./uninstall.sh


# Worked examples
After installation, to run the worked examples

&nbsp;&nbsp;&nbsp;&nbsp; cd examples

&nbsp;&nbsp;&nbsp;&nbsp; mpirun -n [number of cores] python3 [exampleName].py