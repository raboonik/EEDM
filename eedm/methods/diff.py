'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: Functions to compute partial derivatives

'''

import numpy as np
import scipy as spy

def partial(func, axis, spacing):
    fshape = func.shape
    ndim = len(fshape)
    if   ndim == 1:
        spl = spy.interpolate.splrep(spacing,func)      # smoothing
        out = spy.interpolate.splev(spacing,spl,der=1)  # use those knots to get second derivative
    else:
        out  = np.zeros(fshape)
        
        if   ndim == 2:
            if   axis == 0:
                for i in range(fshape[1]):
                    spl      = spy.interpolate.splrep(spacing,func[:,i]) 
                    out[:,i] = spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 1:
                for i in range(fshape[0]):
                    spl      = spy.interpolate.splrep(spacing,func[i,:]) 
                    out[i,:] = spy.interpolate.splev(spacing,spl,der=1) 
        elif ndim == 3:
            if   axis == 0:
                for i in range(fshape[1]):
                    for j in range(fshape[2]):
                        spl        = spy.interpolate.splrep(spacing,func[:,i,j]) 
                        out[:,i,j] = spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 1:
                for i in range(fshape[0]):
                    for j in range(fshape[2]):
                        spl        = spy.interpolate.splrep(spacing,func[i,:,j]) 
                        out[i,:,j] = spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 2:
                for i in range(fshape[0]):
                    for j in range(fshape[1]):
                        spl        = spy.interpolate.splrep(spacing,func[i,j,:]) 
                        out[i,j,:] = spy.interpolate.splev(spacing,spl,der=1) 
        elif ndim == 4:
            if   axis == 0:
                for i in range(fshape[1]):
                    for j in range(fshape[2]):
                        for k in range(fshape[3]):
                            spl          = spy.interpolate.splrep(spacing,func[:,i,j,k]) 
                            out[:,i,j,k] = spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 1:
                for i in range(fshape[0]):
                    for j in range(fshape[2]):
                        for k in range(fshape[3]):
                            spl          = spy.interpolate.splrep(spacing,func[i,:,j,k]) 
                            out[i,:,j,k] = spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 2:
                for i in range(fshape[0]):
                    for j in range(fshape[1]):
                        for k in range(fshape[3]):
                            spl          = spy.interpolate.splrep(spacing,func[i,j,:,k]) 
                            out[i,j,:,k] = spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 3:
                for i in range(fshape[0]):
                    for j in range(fshape[1]):
                        for k in range(fshape[2]):
                            spl          = spy.interpolate.splrep(spacing,func[i,j,k,:]) 
                            out[i,j,k,:] = spy.interpolate.splev(spacing,spl,der=1) 
        else:
            raise ValueError("Error: partial is not ready for functions with more than 4 dimensions!")
        
    return out