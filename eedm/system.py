'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: Script to include all the libraries required for the EEDM code
'''

import glob
import os

def create_dir(dir):
    # No shell involved, so paths with spaces or special characters are safe
    try:
        os.makedirs(dir, exist_ok=True)
    except OSError as e:
        raise OSError("Could not create the output directory " + dir + " (" + str(e) + "). Check that settings.datapath is writable.") from e

def fid(path):
    return sorted(glob.glob(path))