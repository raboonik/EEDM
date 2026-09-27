'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    Date  : 2025-4-03
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Parallelized routine to work out the eigenenergy time derivatives (Eq6 of the paper) 
    and the eigenenergies themselves (Eq9).
    
    It works for full nonlinear ideal-MHD with constant gravity along the -z axis.
    
    Code structure:
        EEDM/
            ├── eedm/
            │   ├── __init__.py            loads settings.py, imports the package
            │   ├── __main__.py            entry point (python -m eedm, or the eedm command)
            │   ├── AutoParallelizePy/     MPI domain decomposition (git submodule)
            │   ├── core/
            │   │   ├── EEDM_eq6.py        Equation 6: eigenpowers in 3D
            │   │   └── EEDM_eq9.py        Equation 9: time integrals on 2D slabs
            │   ├── io/
            │   │   ├── reader.py          snapshot selection and reading for all codes
            │   │   ├── extensionLoader.py
            │   │   ├── larexd/            LARE (.sdf via the sdf module, .cfd readers, staggered-grid averaging)
            │   │   ├── mancha/            MANCHA (.h5)
            │   │   └── pencil/            Pencil Code (VAR snapshots, pencil_read)
            │   ├── methods/diff.py        spline derivatives
            │   ├── utils/
            │   │   ├── crop.py            crop frames
            │   │   ├── decomposition.py   output groupings (settings.decomposition)
            │   │   ├── filter.py          characteristic speeds and degenerate-point conventions
            │   │   └── get_energy_sliced.py  Equation 9 slabs (settings.slicingPnts)
            │   ├── context.py             MPI context and validated settings
            │   ├── system.py
            │   ├── decorators.py
            │   └── SI_constants.py
            └── settings.py
'''



import os, sys
import numpy as np
from mpi4py import MPI

from . import system
from . import const
from . import context as ct
from . import decorators

import settings   # loaded by eedm/__init__.py from the directory EEDM is run from

# Initialize parallelization
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈
#      Init Parallel             #◈
ct.comm     = MPI.COMM_WORLD     #◈
ct.size     = ct.comm.Get_size() #◈
ct.rank     = ct.comm.Get_rank() #◈
#◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈◈

# An error raised on only some of the ranks would leave the others waiting forever at the next collective call:
# print it and abort the whole MPI job instead
def _abort_on_error(exc_type, exc, tb):
    sys.__excepthook__(exc_type, exc, tb)
    sys.stderr.flush()
    if ct.size > 1:
        print("\nEEDM stopped on rank " + str(ct.rank) + " because of the error above; aborting all ranks.", file=sys.stderr, flush=True)
        ct.comm.Abort(1)
sys.excepthook = _abort_on_error


# The run header's logo: a wave (a disturbance) split into its eigenmodes, with the reverse (-) and forward (+) mode of each branch
LOGO = r"""
   _____ _____ ____  __  __          ,----------- fast       f-  f+
  | ____| ____|  _ \|  \/  |        /.----------- Alfven     A-  A+
  |  _| |  _| | | | | |\/| |  ~~~~~<------------- slow       s-  s+
  | |___| |___| |_| | |  | |        \'----------- entropy    ent
  |_____|_____|____/|_|  |_|         '----------- div, g
  EigenEnergy Decomposition Method   (Raboonik et al. 2024a, 2024b, 2025)
"""


def axis_length(centres, faces=None):
    '''Length of the domain along one axis: from the cell faces where the code has them (LARE), otherwise the span of the cell
    centres plus half a cell at each end (exact for uniform grids).'''
    if faces is not None and len(faces) >= 2: return float(faces[-1] - faces[0])
    c = np.asarray(centres, dtype=float)
    return float(c[-1] - c[0]) + (0.5 * float(c[1] - c[0] + c[-1] - c[-2]) if len(c) > 1 else 0.0)


def domain_table(grid, faces, invariant, fullGrid=None, fullFaces=None):
    '''
        The run header's domain table, one row per axis: the full snapshot and the analysed domain (after cropping), each as
        number of cells, length, and range of the cell centres. Without the full grid (Equation 9 only runs, which only know
        the grid of the Equation 6 run), only the analysed columns are shown.
    '''
    def cols(c, f):
        c = np.asarray(c, dtype=float)
        return ["%d" % len(c), "%g" % axis_length(c, f), "[%g, %g]" % (c[0], c[-1])]
    rows = []
    for k, a in enumerate("xyz"):
        full = None if fullGrid is None else cols(fullGrid[k], None if fullFaces is None else fullFaces[k])
        if a in invariant:
            ana = "invariant: 1 cell at %s = %g" % (a, float(np.asarray(grid[k])[0]))
        elif fullGrid is not None and len(fullGrid[k]) == len(grid[k]):
            ana = "not cropped"
        else:
            ana = cols(grid[k], None if faces is None else faces[k])
        rows.append((a, full, ana))
    
    sub   = ["cells", "length", "cell centres"]
    width = lambda j, which: max([len(sub[j])] + [len(r[which][j]) for r in rows if isinstance(r[which], list)])
    def block(values, which):
        if isinstance(values, str): return values
        w = [width(j, which) for j in range(3)]
        return "%*s  %-*s  %-*s" % (w[0], values[0], w[1], values[1], w[2], values[2])
    heads = [("full snapshot", 1), ("analysed (after cropping)", 2)] if fullGrid is not None else [("analysed", 2)]
    wcol  = [max(len(block(sub, which)), len(title), max(len(block(r[which], which)) for r in rows)) for title, which in heads]
    line  = lambda cells: "  q | " + " | ".join("%-*s" % (w, c) for w, c in zip(wcol, cells))
    out   = ["    | " + " | ".join(title.center(w) for (title, _), w in zip(heads, wcol)),
             line([block(sub, which) for _, which in heads]).replace("  q |", "  q |"),
             "  --+" + "+".join("-" * (w + 2) for w in wcol)]
    for a, full, ana in rows:
        cells = ([block(full, 1)] if fullGrid is not None else []) + [block(ana, 2)]
        out.append(("  %s | " % a) + " | ".join("%-*s" % (w, c) for w, c in zip(wcol, cells)))
    return [o.rstrip() for o in out]


def print_header(outDirec, dirDict, readObj=None):
    from . import io
    if readObj is not None:
        nSnap = readObj.nt
        faces = None if readObj.xb is None else (readObj.xb, readObj.yb, readObj.zb)
        dom   = domain_table((readObj.xc, readObj.yc, readObj.zc), faces, ct.invariantAxes, readObj.fullGrid, readObj.fullFaces)
    else:
        # Equation 9 only: the grid of the Equation 6 run that produced the files
        nSnap = len(io.reader.select_snapshots()[0])                  # on all ranks, so any error is raised everywhere
        gp    = os.path.join(dirDict["parent"], "grid_params.h5")
        if os.path.isfile(gp):
            import h5py
            with h5py.File(gp, "r") as h:
                faces = (h["xb"][()], h["yb"][()], h["zb"][()]) if "xb" in h else None
                inv   = [a for a in str(h.attrs.get("invariantAxes", "")).split(",") if a]
                dom   = domain_table((h["xc"][()], h["yc"][()], h["zc"][()]), faces, inv)
        else:
            dom = ["not known yet (no grid_params.h5 from an Equation 6 run)"]
    if ct.rank != ct.mainrank: return
    first, last = settings.dataInterval
    task  = {"eq6": "Equation 6 only", "eq9": "Equation 9 only"}.get(settings.computationSwitch, "Equations 6 and 9")
    units = "SI (mu0 = %.6g)" % const.mu0 if settings.dimensionality_switch else "non-dimensional (mu0 = 1)"
    grav  = ("g = %g along -%s" % (settings.g, ct.verticalAxis) if settings.g > 0 else "none (g = 0)") + \
            ("" if ct.verticalAxis == "z" else " (vertical axis %s: rotated to EEDM's z internally, outputs in the simulation's axes)" % ct.verticalAxis)
    print(LOGO)
    for label, value in (("simulation code", settings.simCode + " (." + settings.dataExt + " snapshots)"),
                         ("data"           , settings.datapath),
                         ("snapshots"      , "%d selected (dataInterval = [%s, %s], skip = %s)" % (nSnap, first, last, settings.skip)),
                         ("dimensions"     , "3D" if not ct.invariantAxes else "%dD (invariant along %s)" % (ct.dimensions, " and ".join(ct.invariantAxes))),
                         ("domain (%s)" % ("m" if settings.dimensionality_switch else "code units"), None),
                         ("MPI ranks"      , ct.size),
                         ("units"          , units),
                         ("gravity"        , grav),
                         ("gamma"          , "%g" % settings.gamma),
                         ("precision"      , ct.dtype.name),
                         ("decomposition"  , ct.decomposition),
                         ("computing"      , task),
                         ("output"         , outDirec)):
        if value is None:                                             # the domain table, as a block under its label
            print("  %s:" % label)
            for row in dom: print("      " + row)
        else:
            print("  %-19s: %s" % (label, value))
    print("", flush=True)
    for note in ([] if readObj is None else readObj.notes): print(note + "\n", flush=True)


@decorators.timeit
def main():
    if not settings.dimensionality_switch: const.mu0  = 1
    
    datapath    = settings.datapath
    outDirecL1  = os.path.join(datapath  , 'EEDM_results')
    outDirecSp  = os.path.join(outDirecL1, 'Char_Speeds')
    outDirecEn  = os.path.join(outDirecL1, 'Predecomposition_Energies')
    outDirecEx  = os.path.join(outDirecL1, 'Extras')
    
    outDirecL2  = os.path.join(outDirecL1, 'Decomposed_EigenEnergies')
    outDirecEq6 = os.path.join(outDirecL2, 'PaperIII_Eq_6_3D')
    outDirecEq9 = os.path.join(outDirecL2, 'PaperIII_Eq_9_2DSlabs')
    
    dirDict = {"parent": outDirecL1, "eq6": outDirecEq6, "eq9": outDirecEq9, "speed": outDirecSp, "energy": outDirecEn, "extra": outDirecEx}
    
    if   settings.computationSwitch == "eq6": ct.eq9Cond = False
    elif settings.computationSwitch == "eq9": ct.eq6Cond = False
    
    # The reader (grid and cropping) is set up once, here, so that the run header can show the domain; Equation 6 then uses it
    from . import io
    readObj = io.reader.EEDM_reader() if ct.eq6Cond else None
    
    print_header(outDirecL1, dirDict, readObj)
    
    # Create the output directories
    if ct.rank == ct.mainrank:
        # Level 1
        system.create_dir(outDirecL1)
        system.create_dir(outDirecL2)
        
        # Level 2
        system.create_dir(outDirecEq6)
        system.create_dir(outDirecEq9)
        
        # Conditional levels
        if ct.SpCond             : system.create_dir(outDirecSp)
        if ct.EnCond             : system.create_dir(outDirecEn)
        if ct.DbCond or ct.PkCond: system.create_dir(outDirecEx)
    
    ct.comm.barrier()
    
    if ct.eq6Cond:
        from .core import EEDM_eq6
        EEDM_eq6.run(dirDict, readObj)
    
    ct.comm.barrier()
    
    if ct.eq9Cond:
        from .core import EEDM_eq9
        EEDM_eq9.run(dirDict)
    
    ct.comm.barrier()
    
    if ct.rank == ct.mainrank: print("All done! The results are in " + outDirecL1 + "\n")


if __name__ == "__main__":
    main()