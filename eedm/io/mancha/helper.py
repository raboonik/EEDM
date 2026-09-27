import numpy as np

def remove_extra_files(snapLst):
    '''
        Split the file list into the main snapshots and their "<snapshot>_extra.h5" files
        (newer Mancha versions store the pressure there). The returned extra list is aligned
        with the snapshot list (extraLst[i] belongs to snapLst[i]), or empty if there are none.
    '''
    # Remove the background file
    snapLst  = [f for f in snapLst if 'background_plasma.h5' not in f]

    extraLst = [f for f in snapLst if '_extra.h5'     in f]
    snapLst  = [f for f in snapLst if '_extra.h5' not in f]

    if len(extraLst) > 0:
        pairs   = [f[:-len('.h5')] + '_extra.h5' for f in snapLst]
        missing = [f for f in pairs if f not in extraLst]
        if len(missing) > 0:
            raise ValueError("Found Mancha _extra.h5 files, but not one for every snapshot (" + str(len(missing)) + " missing, e.g. " + missing[0] +
                             "). Provide the _extra.h5 file of every snapshot, or none.")
        extraLst = pairs

    return snapLst, extraLst


def get_grid(dat):
    dx = dat.attrs['cellsize'][0]
    dy = dat.attrs['cellsize'][1]
    dz = dat.attrs['cellsize'][2]
    nx = dat.attrs['metadata'][1]
    ny = dat.attrs['metadata'][2]
    nz = dat.attrs['metadata'][3]

    # Build from the index: np.arange(0, dx*nx, dx) can return nx+1 points due to round-off
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    
    return x, y, z