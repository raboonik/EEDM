from libs import prerequisites as pre

def partial(func, axis, spacing):
    fshape = func.shape
    ndim = len(fshape)
    if   ndim == 1:
        spl = pre.spy.interpolate.splrep(spacing,func)      # smoothing
        out = pre.spy.interpolate.splev(spacing,spl,der=1)  # use those knots to get second derivative
    else:
        out  = pre.np.zeros(fshape)
        
        if   ndim == 2:
            if   axis == 0:
                for i in range(fshape[1]):
                    spl      = pre.spy.interpolate.splrep(spacing,func[:,i]) 
                    out[:,i] = pre.spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 1:
                for i in range(fshape[0]):
                    spl      = pre.spy.interpolate.splrep(spacing,func[i,:]) 
                    out[i,:] = pre.spy.interpolate.splev(spacing,spl,der=1) 
        elif ndim == 3:
            if   axis == 0:
                for i in range(fshape[1]):
                    for j in range(fshape[2]):
                        spl        = pre.spy.interpolate.splrep(spacing,func[:,i,j]) 
                        out[:,i,j] = pre.spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 1:
                for i in range(fshape[0]):
                    for j in range(fshape[2]):
                        spl        = pre.spy.interpolate.splrep(spacing,func[i,:,j]) 
                        out[i,:,j] = pre.spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 2:
                for i in range(fshape[0]):
                    for j in range(fshape[1]):
                        spl        = pre.spy.interpolate.splrep(spacing,func[i,j,:]) 
                        out[i,j,:] = pre.spy.interpolate.splev(spacing,spl,der=1) 
        elif ndim == 4:
            if   axis == 0:
                for i in range(fshape[1]):
                    for j in range(fshape[2]):
                        for k in range(fshape[3]):
                            spl          = pre.spy.interpolate.splrep(spacing,func[:,i,j,k]) 
                            out[:,i,j,k] = pre.spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 1:
                for i in range(fshape[0]):
                    for j in range(fshape[2]):
                        for k in range(fshape[3]):
                            spl          = pre.spy.interpolate.splrep(spacing,func[i,:,j,k]) 
                            out[i,:,j,k] = pre.spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 2:
                for i in range(fshape[0]):
                    for j in range(fshape[1]):
                        for k in range(fshape[3]):
                            spl          = pre.spy.interpolate.splrep(spacing,func[i,j,:,k]) 
                            out[i,j,:,k] = pre.spy.interpolate.splev(spacing,spl,der=1) 
            elif axis == 3:
                for i in range(fshape[0]):
                    for j in range(fshape[1]):
                        for k in range(fshape[2]):
                            spl          = pre.spy.interpolate.splrep(spacing,func[i,j,k,:]) 
                            out[i,j,k,:] = pre.spy.interpolate.splev(spacing,spl,der=1) 
        else:
            raise ValueError("Error: partial is not ready for functions with more than 4 dimensions!")
        
    return out