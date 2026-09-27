'''
    The simulation frame and the EEDM frame. EEDM's equations take gravity along -z. When the simulation's vertical
    (settings.verticalAxis, the axis pointing up) is x or y, the data are rotated into the EEDM frame when they are read,
    and every output is rotated back, so that users only ever see the simulation's own axes. The rotation is a cyclic
    permutation of the axes (a proper rotation, so the frame stays right-handed and nothing is flipped):

        verticalAxis   EEDM (x, y, z) =
        "z"            simulation (x, y, z)    (identity)
        "y"            simulation (z, x, y)
        "x"            simulation (y, z, x)

    Arrays are (x, y, z) ordered in both frames. PERM[k] is the simulation axis that becomes EEDM axis k.
'''

import numpy as np

from .. import context as ct

PERM     = ct.PERMUTATIONS[ct.verticalAxis]
INV      = tuple(int(i) for i in np.argsort(PERM))
IDENTITY = PERM == (0, 1, 2)


def sim_axis(q):
    '''Simulation axis letter of the EEDM axis q ("x", "y", or "z").'''
    return "xyz"[PERM["xyz".index(q)]]


def eedm_axis(a):
    '''EEDM axis letter of the simulation axis a.'''
    return "xyz"[INV["xyz".index(a)]]


def array_in(a):
    '''A 3D (x, y, z) array from the simulation frame to the EEDM frame.'''
    return a if IDENTITY else np.transpose(a, PERM)


def array_out(a):
    '''A 3D (x, y, z) array from the EEDM frame back to the simulation frame (arrays of size 1, the placeholders, pass unchanged).'''
    return a if IDENTITY or np.ndim(a) != 3 else np.transpose(a, INV)


def vector_in(vx, vy, vz):
    '''The components of a vector field from the simulation frame to the EEDM frame (each component also rotated as an array).'''
    comps = (vx, vy, vz)
    return tuple(array_in(comps[PERM[k]]) for k in range(3))
