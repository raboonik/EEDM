'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: Functions to aid in reading simulation data and working out 
                 the subgrid associated with the cropping frame 
'''

import numpy as np

smallestAlfvSpeedAllowed = 1.e-30
smallestWaveFieldAllowed = 1.e-30
epsilon                  = 9.e-15

def SQRTnegFilter(dat):
    if np.min(dat) < 0:
        mask = dat < 0
        if np.abs(np.max(dat[mask]) - np.min(dat[mask])) < 1.e-9:
            dat[mask] = 0.
    else:
        pass



