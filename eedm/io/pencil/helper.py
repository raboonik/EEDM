'''
    Helpers to feed Pencil Code snapshots (data/allprocs/VARN) to EEDM_reader,
    using the lazy reader in pencil_read.py.

    Pencil stores all variables on the same (collocated) grid, so there is no
    staggering to undo. Ghost zones are removed by pencil_read.

    Units:
        dimensionality_switch = True : everything is converted to SI with pencil_read's
                             units_SI, so it is consistent with const.mu0 (SI).
        dimensionality_switch = False: Pencil code units, except that B is rescaled by
                             1/sqrt(mu0_code). The core sets mu0 = 1 for
                             non-dimensional data, whereas Pencil's code-unit
                             mu0 is param.mu0 (not necessarily 1); rescaling B
                             keeps B^2/mu0 and the Alfven speed unchanged.
'''
import os

import numpy as np

from . import pencil_read


def open_run(datapath):
    return pencil_read.Reader(datapath)


def get_feed(datapath, snapshots):
    '''
        One label per snapshot: datapath/VAR<zero-padded N>.var. These are labels,
        not files. The core names its outputs after the label without its extension
        (e.g. EigenenergyDDT_VAR000500.h5), and eq9 integrates in time over the outputs
        in sorted order, so the index is zero-padded to sort in time order.
    '''
    width = max(6, len(str(max(snapshots))))
    return [os.path.join(datapath, "VAR" + str(n).zfill(width) + ".var") for n in snapshots]


def mu0_code(run):
    return getattr(run.param, "mu0", 1.0)


def get_grid(run, dimensional):
    L = run.units_SI.unit_length if dimensional else 1.0
    return run.grid.x * L, run.grid.y * L, run.grid.z * L


def get_data(run, ivar, framec, dimensional):
    '''
        Read one snapshot and return (time, rho, vx, vy, vz, bx, by, bz, p), cropped to framec.
        Fields are read one group at a time and the reader's cache is cleared afterwards,
        so only the cropped arrays stay in memory.
    '''
    run.ivar = ivar
    sl = (slice(framec[0], framec[1]), slice(framec[2], framec[3]), slice(framec[4], framec[5]))

    if dimensional:
        u      = run.units_SI
        fRho   = u.unit_density
        fV     = u.unit_velocity
        fB     = u.unit_magnetic
        fP     = u.unit_pressure
        fT     = u.unit_time
    else:
        fRho, fV, fP, fT = 1.0, 1.0, 1.0, 1.0
        fB = 1.0 / np.sqrt(mu0_code(run))

    vx, vy, vz = [q[sl] * fV for q in (run.ux, run.uy, run.uz)]  # a new array, not a view
    time = run.time * fT   # cached with the velocities, so no extra read (asking first would read lnrho just for t)
    run.clear()
    bx, by, bz = [q[sl] * fB for q in (run.bx, run.by, run.bz)]
    run.clear()
    rho = run.rho[sl] * fRho
    p   = run.pp[sl]  * fP
    run.clear()

    return time, rho, vx, vy, vz, bx, by, bz, p


def check_gravity(run, g, dimensional, vertical="z"):
    '''
        Compare settings.g with the run's constant gravity along the vertical axis (Pencil: e.g. gravz < 0 points along -z).
        Returns a warning string, or None if they agree or the run has no constant gravity along that axis.
    '''
    param = run.param
    name  = "grav" + vertical
    if not hasattr(param, name) or str(getattr(param, name + "_profile", "")).strip() != "const":
        return None
    gRun = -getattr(param, name)
    if dimensional:
        u    = run.units_SI
        gRun = gRun * u.unit_length / u.unit_time**2
    if np.isclose(g, gRun, rtol=1e-3, atol=0):
        return None
    unit = "m/s^2" if dimensional else "code units"
    return "Warning: settings.g = " + str(g) + ", but the Pencil run has uniform gravity of magnitude " + str(abs(gRun)) + " " + unit + \
           " pointing along " + ("-" if gRun > 0 else "+") + vertical + " (" + name + " = " + str(getattr(param, name)) + " in param.nml). EEDM uses settings.g.\n"
