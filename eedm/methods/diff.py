'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: Functions to compute partial derivatives

'''

import numpy as np
import scipy as spy

try:
    # Vectorised along any axis (SciPy >= 0.19)
    from scipy.interpolate import make_interp_spline
except ImportError:
    make_interp_spline = None

def partial(func, axis, spacing):
    '''
        First derivative of func along axis, from the interpolating cubic spline through
        the data (not-a-knot end conditions). Uses make_interp_spline when available and
        falls back to the per-line splrep/splev loops otherwise; both give the same spline.
    '''
    if func.shape[axis] == 1:
        # An invariant axis of a 2D or 1D run (settings.invariantAxes), reduced to one cell by the reader: d/dq = 0
        return np.zeros_like(func)
    if make_interp_spline is not None:
        # check_finite=False: NaNs propagate to the output (as with splrep) instead of raising
        out = make_interp_spline(spacing, func, k=3, axis=axis, check_finite=False).derivative()(spacing)
    else:
        out = _partial_legacy(func, axis, spacing)
    # The spline is fitted in float64; return the derivative in the precision of the input
    return out.astype(func.dtype, copy=False)

def _partial_legacy(func, axis, spacing):
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
            raise ValueError("partial() supports arrays with up to 4 dimensions, not " + str(len(fshape)) + ".")
        
    return out