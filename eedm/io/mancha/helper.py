import numpy as np

def remove_extra_files(snapLst):
    # Remove the background file
    snapLstWiex = [f for f in snapLst if 'background_plasma.h5' not in f]
    
    cond = np.any(['_extra.h5' in f for f in snapLstWiex])
    
    snapLstWoex = []
    if cond:
        snapLstWiex = [f for f in snapLstWiex if '_extra.h5' not in f]
        snapLstWoex = [f for f in snapLstWiex if '_extra.h5'     in f]
    else:
        pass
        
    return snapLstWiex, snapLstWoex


def get_grid(dat):
    dx = dat.attrs['cellsize'][0]
    dy = dat.attrs['cellsize'][1]
    dz = dat.attrs['cellsize'][2]
    nx = dat.attrs['metadata'][1]
    ny = dat.attrs['metadata'][2]
    nz = dat.attrs['metadata'][3]

    x = np.arange(0., dx*nx, dx)
    y = np.arange(0., dy*ny, dy)
    z = np.arange(0., dz*nz, dz)
    
    return x, y, z