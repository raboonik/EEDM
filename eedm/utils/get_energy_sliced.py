'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com

    Github: https://github.com/raboonik

    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917

    Description: Functions to aid in working out the 2D slices used in the
                 computation of Equation 9
'''

import os
import numbers
import numpy as np

# The axis normal to each slicing plane
NORMAL = {"xy": "z", "xz": "y", "yz": "x"}


def resolve_slicing_points(plane, points, kind, xc, yc, zc):
    '''
        Turn settings.slicingPnts into the grid indices of the Equation 9 slabs along the plane's normal axis.
            kind = "coordinate": positions in the data's length unit, each snapped to the nearest grid point
            kind = "index"     : grid indices, counted from the start of the (cropped) grid
        xc, yc, zc are the (cropped) cell-centred grid. Returns (indices, coordinates, notes): the slab indices
        without repeats, in the order given, their coordinates, and notes on points merged into one slab.
        Raises ValueError with an explanation for anything that cannot be used.
    '''
    if plane not in NORMAL:
        raise ValueError("settings.slicingPlane = " + repr(plane) + ' is not supported. Use "xy", "xz", or "yz".')
    if kind is None:
        raise ValueError('settings.py has no "slicingPntsType" setting, which says how to read slicingPnts. Add slicingPntsType = "coordinate" ' +
                         '(positions in the data\'s length unit) or "index" (grid indices of the cropped grid).')
    if kind not in ("coordinate", "index"):
        raise ValueError("settings.slicingPntsType = " + repr(kind) + ' is not supported. Use "coordinate" (positions in the data\'s length unit) ' +
                         'or "index" (grid indices of the cropped grid).')
    normal = NORMAL[plane]
    axis   = np.asarray({"x": xc, "y": yc, "z": zc}[normal], dtype=float)
    n      = len(axis)

    values = list(np.asarray(points, dtype=object).ravel())
    if len(values) == 0:
        raise ValueError("settings.slicingPnts is empty, so there is no slab to compute Equation 9 on. Give at least one " +
                         ("position" if kind == "coordinate" else "grid index") + " along " + normal + ".")
    if not all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in values):
        raise ValueError("settings.slicingPnts must contain numbers only, not " + repr(points) + ".")

    if kind == "index":
        bad = [v for v in values if not isinstance(v, numbers.Integral)]
        if bad:
            raise ValueError('settings.slicingPntsType = "index", but settings.slicingPnts contains ' + ", ".join(repr(v) for v in bad) +
                             '. Grid indices must be integers (e.g. 192, not 192.0); for positions, set slicingPntsType = "coordinate".')
        bad = [int(v) for v in values if not 0 <= v < n]
        if bad:
            raise ValueError("settings.slicingPnts contains the grid index(es) " + ", ".join(str(v) for v in bad) + ", outside the " + str(n) +
                             " grid points along " + normal + " (valid: 0 to " + str(n - 1) + ", counted from the start of the cropped grid).")
        wanted = [int(v) for v in values]
    else:
        # Accept points up to half a grid spacing beyond the end points, which still snap to the edge of the grid
        dlo    = 0.5 * abs(axis[1]  - axis[0])  if n > 1 else 0.0
        dhi    = 0.5 * abs(axis[-1] - axis[-2]) if n > 1 else 0.0
        lo, hi = min(axis[0], axis[-1]) - dlo, max(axis[0], axis[-1]) + dhi
        bad    = [float(v) for v in values if not lo <= float(v) <= hi]
        if bad:
            raise ValueError("settings.slicingPnts contains the position(s) " + ", ".join("%g" % v for v in bad) + ", outside the (cropped) domain, " +
                             "where " + normal + " runs from %g to %g." % (axis[0], axis[-1]))
        wanted = [int(np.abs(axis - float(v)).argmin()) for v in values]

    # Merge points that land on the same grid point, keeping the order given
    indices, notes = [], []
    for idx in dict.fromkeys(wanted):
        same = [v for v, w in zip(values, wanted) if w == idx]
        if len(same) > 1:
            if kind == "coordinate":
                notes.append("Note: the slicing points " + ", ".join("%s = %g" % (normal, float(v)) for v in same) +
                             " map to the same grid point, index %d (%s = %g); computing one slab there." % (idx, normal, axis[idx]))
            else:
                notes.append("Note: grid index %d (%s = %g) is listed %d times; computing one slab there." % (idx, normal, axis[idx], len(same)))
        indices.append(idx)
    return np.array(indices, dtype=int), [float(axis[i]) for i in indices], notes


def get_slice_2d(axis, nq0, xc,yc,zc,localOutpath, slq1=0, elq1=0, slq2=0, elq2=0):
    # nq0: the slab indices along the plane's normal axis, as returned by resolve_slicing_points
    nx = len(xc)
    ny = len(yc)
    nz = len(zc)

    # Assume the single-core version
    if elq1 == 0:
        if   axis == "xy":
            elq1 = nx
            elq2 = ny
        elif  axis == "xz":
            elq1 = nx
            elq2 = nz
        elif  axis == "yz":
            elq1 = ny
            elq2 = nz

    nq0  = [int(i) for i in np.asarray(nq0).ravel()]
    lenq = len(nq0)

    if   axis == "xy":
        nq1   = nx
        nq2   = ny
        axis1 = xc
        axis2 = yc
        axis3 = zc
        nq1i         = [slq1 for i in range(lenq)]
        nq1f         = [elq1 for i in range(lenq)]
        nq2i         = [slq2 for i in range(lenq)]
        nq2f         = [elq2 for i in range(lenq)]
        nq3i         = nq0
        nq3f         = [i+1 for i in nq0]
        outfilename1 = [os.path.join(localOutpath, "EE_Slab_"+axis+"_nz0_"+str(nq0[i])+".h5") for i in range(lenq)]
        outfilename2 = [os.path.join(localOutpath, "EE_Slab_"+axis+"_vars_nz0_"+str(nq0[0])+".h5") for i in range(lenq)]
    elif axis == "xz":
        nq1  = nx
        nq2  = nz
        axis1 = xc
        axis2 = zc
        axis3 = yc
        nq1i = [slq1 for i in range(lenq)]
        nq1f = [elq1 for i in range(lenq)]
        nq2i = nq0
        nq2f = [i+1 for i in nq0]
        nq3i = [slq2 for i in range(lenq)]
        nq3f = [elq2 for i in range(lenq)]
        outfilename1 = [os.path.join(localOutpath, "EE_Slab_"+axis+"_ny0_"+str(nq0[i])+".h5") for i in range(lenq)]
        outfilename2 = [os.path.join(localOutpath, "EE_Slab_"+axis+"_vars_ny0_"+str(nq0[0])+".h5") for i in range(lenq)]
    elif axis == "yz":
        nq1  = ny
        nq2  = nz
        axis1 = yc
        axis2 = zc
        axis3 = xc
        nq1i = nq0
        nq1f = [i+1 for i in nq0]
        nq2i = [slq1 for i in range(lenq)]
        nq2f = [elq1 for i in range(lenq)]
        nq3i = [slq2 for i in range(lenq)]
        nq3f = [elq2 for i in range(lenq)]
        outfilename1 = [os.path.join(localOutpath, "EE_Slab_"+axis+"_nx0_"+str(nq0[i])+".h5") for i in range(lenq)]
        outfilename2 = [os.path.join(localOutpath, "EE_Slab_"+axis+"_vars_nx0_"+str(nq0[0])+".h5") for i in range(lenq)]
    else:
        raise ValueError("settings.slicingPlane = " + repr(axis) + ' is not supported. Use "xy", "xz", or "yz".')

    nq0Loc = [axis3[nq00] for nq00 in nq0]

    return nx,ny,nz,axis1,axis2,axis3,nq0, nq0Loc, lenq, nq1, nq2, nq1i, nq1f, nq2i, nq2f, nq3i, nq3f, outfilename1, outfilename2
