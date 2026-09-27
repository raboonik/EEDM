"""
Lazy reader for Pencil Code snapshots and grid data.

Nothing is read at import time or when the reader is created. Each quantity
is read from disk the first time it is accessed, then cached.

All fields, the grid and the time are returned in (dimensionless) code units.
To convert, multiply by the matching factor from `units` (cgs) or `units_SI`:

>>> import pencil_read
>>> d = pencil_read('/somepath/data/allprocs', ivar=500)  # or pencil_read.Reader(...)
>>> grid = d.grid                               # x, y, z, ... (ghost zones removed)
>>> ux = d.ux                                   # x-velocity, shape (nx, ny, nz)
>>> ux_cgs = d.ux * d.units.unit_velocity       # cm/s
>>> bz_SI = d.bz * d.units_SI.unit_magnetic     # T
>>> x_SI = d.grid.x * d.units_SI.unit_length    # m
>>> d.ivar = 501                                # switch snapshot (clears cached fields)

Conversion factor for each quantity:

    ux, uy, uz                    unit_velocity
    bx, by, bz                    unit_magnetic
    rho                           unit_density
    TT                            unit_temperature
    pp                            unit_pressure
    time, grid.t                  unit_time
    grid.x/y/z, dx/dy/dz, Lx/y/z  unit_length
    grid.dx_1, grid.dx_tilde, ... 1 / unit_length

If ivar is omitted, the latest VARN file in data/allprocs is used.
"""

import os
import re
import sys
import types
from types import SimpleNamespace

import numpy as np

__all__ = ["Reader"]

# Physical constants (cgs)
K_B_CGS = 1.3806505e-16  # erg/K
M_U_CGS = 1.66053886e-24  # g
M_E_CGS = 9.10938188e-28  # g
HBAR_CGS = 1.054571596e-27  # erg s
EV_CGS = 1.602e-12  # erg

# Multiply a cgs value by these to get SI
CGS_TO_SI = {
    "unit_length": 1e-2,  # cm -> m
    "unit_velocity": 1e-2,  # cm/s -> m/s
    "unit_density": 1e3,  # g/cm^3 -> kg/m^3
    "unit_temperature": 1.0,  # K -> K
    "unit_magnetic": 1e-4,  # G -> T
    "unit_mass": 1e-3,  # g -> kg
    "unit_energy": 1e-7,  # erg -> J
    "unit_time": 1.0,  # s -> s
    "unit_pressure": 1e-1,  # dyn/cm^2 -> Pa
}


def _datadir(datapath):
    """Return the Pencil 'data' directory from a path to data/ or data/allprocs."""
    path = os.path.abspath(os.path.expanduser(os.fspath(datapath))).rstrip(os.sep)
    if os.path.basename(path) == "allprocs":
        path = os.path.dirname(path)
    if not os.path.isfile(os.path.join(path, "dim.dat")):
        raise FileNotFoundError(f"No dim.dat in {path}, so it is not a Pencil Code data directory. "
                                f"Point to the run's data/ or data/allprocs/ directory.")
    return path


class Reader:
    """
    Lazy reader for one Pencil Code snapshot. Everything is in code units.

    Parameters
    ----------
    datapath : str or Path
        Path to data/allprocs (or to data/).
    ivar : int, optional
        Index of the VAR file to read. Default: the latest one in allprocs.
    """

    def __init__(self, datapath, ivar=None):
        self.datadir = _datadir(datapath)
        self._cache = {}  # snapshot-dependent quantities
        self._static = {}  # grid, param, dim, units
        self._ivar = None
        self.ivar = ivar if ivar is not None else self.snapshots[-1]

    def __repr__(self):
        return f"pencil_read.Reader('{self.datadir}', ivar={self.ivar})"

    # ------------------------------------------------------------------
    # Snapshot selection
    # ------------------------------------------------------------------
    @property
    def snapshots(self):
        """Sorted list of available VAR indices in data/allprocs."""
        allprocs = os.path.join(self.datadir, "allprocs")
        found = [int(m.group(1)) for f in os.listdir(allprocs) if (m := re.fullmatch(r"VAR(\d+)", f))]
        if not found:
            raise FileNotFoundError(f"No VAR<N> snapshots found in {allprocs}.")
        return sorted(found)

    @property
    def ivar(self):
        return self._ivar

    @ivar.setter
    def ivar(self, value):
        value = int(value)
        if not os.path.isfile(os.path.join(self.datadir, "allprocs", f"VAR{value}")):
            avail = self.snapshots
            raise FileNotFoundError(f"VAR{value} not found in {os.path.join(self.datadir, 'allprocs')} "
                                    f"({len(avail)} snapshots available, VAR{avail[0]} to VAR{avail[-1]}).")
        if value != self._ivar:
            self._ivar = value
            self._cache.clear()

    def clear(self):
        """Drop all cached snapshot fields to free memory."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Static data (read once, shared by all snapshots)
    # ------------------------------------------------------------------
    def _static_get(self, key, loader):
        if key not in self._static:
            self._static[key] = loader()
        return self._static[key]

    @property
    def param(self):
        """Simulation parameters (pencil Param object)."""
        import pencil as pc

        return self._static_get("param", lambda: pc.read.param(datadir=self.datadir, quiet=True))

    @property
    def dim(self):
        """Grid dimensions (pencil Dim object)."""
        import pencil as pc

        return self._static_get("dim", lambda: pc.read.dim(datadir=self.datadir))

    @property
    def pc_grid(self):
        """Raw pencil Grid object in code units, including ghost zones."""
        import pencil as pc

        return self._static_get("pc_grid", lambda: pc.read.grid(datadir=self.datadir, precision="d"))

    @property
    def grid(self):
        """
        All fields of the pencil grid in code units, with ghost zones removed
        from the per-axis arrays (x, dx_1, dx_tilde, ...).
        """

        def load():
            g, dm = self.pc_grid, self.dim
            sl = {"x": slice(dm.l1, dm.l2 + 1), "y": slice(dm.m1, dm.m2 + 1), "z": slice(dm.n1, dm.n2 + 1)}
            out = SimpleNamespace()
            for a in "xyz":
                s = sl[a]
                setattr(out, a, getattr(g, a)[s])
                setattr(out, f"d{a}_1", getattr(g, f"d{a}_1")[s])
                setattr(out, f"d{a}_tilde", getattr(g, f"d{a}_tilde")[s])
            for a in "xyz":
                setattr(out, f"d{a}", getattr(g, f"d{a}"))
            for a in "xyz":
                setattr(out, f"L{a}", getattr(g, f"L{a}"))
            out.t = g.t
            return out

        return self._static_get("grid", load)

    def _base_units_cgs(self):
        """Pencil's unit_* parameters, converted to cgs if the run used SI."""
        p = self.param
        names = ["unit_length", "unit_velocity", "unit_density", "unit_temperature", "unit_magnetic"]
        system = str(getattr(p, "unit_system", "cgs")).strip().lower()
        if system == "cgs":
            return {n: getattr(p, n) for n in names}
        if system == "si":
            return {n: getattr(p, n) / CGS_TO_SI[n] for n in names}
        raise ValueError(f"param.nml has unit_system = {p.unit_system!r}, so the units of the unit_* parameters are unknown "
                         f"(expected 'cgs' or 'SI').")

    @property
    def units(self):
        """
        Code -> cgs conversion factors (multiply code data by these), plus
        physical constants expressed in code units (used for the pressure).
        """

        def load():
            u = SimpleNamespace(**self._base_units_cgs())
            u.unit_mass = u.unit_density * u.unit_length**3
            u.unit_energy = u.unit_mass * u.unit_velocity**2
            u.unit_time = u.unit_length / u.unit_velocity
            u.unit_pressure = u.unit_density * u.unit_velocity**2

            # Constants in code units (needed for the pressure)
            p = self.param
            u.k_B = K_B_CGS * u.unit_temperature / u.unit_energy
            u.m_u = M_U_CGS / u.unit_mass
            u.ion_pot = p.chih_ev * EV_CGS / u.unit_energy
            u.me = M_E_CGS / u.unit_mass
            u.hbar = HBAR_CGS / (u.unit_energy * u.unit_time)
            u.Rgas = u.k_B / u.m_u
            u.xHe = p.xhe
            u.rho_e = (1.0 + 4.0 * u.xHe) * u.m_u * (u.me * u.ion_pot / (2.0 * np.pi * u.hbar**2)) ** 1.5
            u.mu1_0 = 1.0 / (1.0 + 4.0 * u.xHe)
            return u

        return self._static_get("units", load)

    @property
    def units_SI(self):
        """Code -> SI conversion factors (multiply code data by these)."""
        return self._static_get(
            "units_SI", lambda: SimpleNamespace(**{n: getattr(self.units, n) * f for n, f in CGS_TO_SI.items()})
        )

    # ------------------------------------------------------------------
    # Snapshot data (code units, ghost zones removed)
    # ------------------------------------------------------------------
    def _read_var(self, var_list):
        """Read the given variables from the current snapshot (with ghost zones)."""
        import pencil as pc

        return pc.read.varraw(ivar=self.ivar, datadir=self.datadir, var_list=var_list, quiet=True)

    def _trim(self, arr):
        dm = self.dim
        return arr[dm.l1 : dm.l2 + 1, dm.m1 : dm.m2 + 1, dm.n1 : dm.n2 + 1]

    def _get(self, key, loader):
        if key not in self._cache:
            self._cache.update(loader())
        return self._cache[key]

    def _load_uu(self):
        var = self._read_var(["uu"])
        return {"ux": self._trim(var.ux), "uy": self._trim(var.uy), "uz": self._trim(var.uz), "t": var.t}

    def _load_rho(self):
        var = self._read_var(["lnrho"])
        return {"rho": np.exp(self._trim(var.lnrho)), "t": var.t}

    def _load_TT(self):
        var = self._read_var(["lnTT"])
        return {"TT": np.exp(self._trim(var.lnTT)), "t": var.t}

    def _check_b_ext(self):
        """B is rebuilt here as curl(A) + a constant, uniform B_ext. Refuse runs whose field Pencil builds differently."""
        p = self.param
        used = [(name, why) for name, value, why in (
            ("omega_Bz_ext",       getattr(p, "omega_bz_ext", 0.0),          "B_ext precesses about z in time"),
            ("lbext_moving_layer", getattr(p, "lbext_moving_layer", False),  "B_ext is confined to a moving layer"),
            ("lhubble_magnetic",   getattr(p, "lhubble_magnetic", False) and np.any(np.asarray(p.b_ext) != 0),
                                                                             "B_ext is scaled by the cosmological scale factor"),
            ("ladd_global_field",  getattr(p, "ladd_global_field", False),   "a spatially varying external field is stored as global arrays"),
            ("dipole_moment",      getattr(p, "dipole_moment", 0.0),         "a dipole field is added"),
            ("B0_ext_z",           getattr(p, "b0_ext_z", 0.0),              "a stratified Bz(z) is added"),
        ) if value]
        if used:
            raise NotImplementedError(
                "This Pencil run adds an external magnetic field that pencil_read cannot reproduce (it only adds a constant, "
                "uniform B_ext): " + "; ".join(f"{name} is set, so {why}" for name, why in used) + ".")

    def _load_bb(self):
        from pencil.math.derivatives import curl as curl_pc

        self._check_b_ext()
        var = self._read_var(["aa"])
        # Pencil derivatives expect (3, z, y, x); data is stored as (x, y, z, 3)
        bb = curl_pc(np.transpose(var.aa), grid=self.pc_grid)
        # B = curl(A) + B_ext, with the uniform external field added to all three components (as in Pencil's magnetic.f90)
        bb += np.asarray(self.param.b_ext, dtype=bb.dtype)[:, None, None, None]
        dm = self.dim
        bb = bb[:, dm.n1 : dm.n2 + 1, dm.m1 : dm.m2 + 1, dm.l1 : dm.l2 + 1]
        return {"bx": bb[0].T, "by": bb[1].T, "bz": bb[2].T, "t": var.t}

    def _load_pp(self):
        """Pressure, including partial hydrogen ionisation (Saha equation)."""
        u = self.units
        TT, rho = self.TT, self.rho
        kT = u.k_B * TT
        q = (u.rho_e / rho) * (u.ion_pot / kT) ** (-1.5) * np.exp(-u.ion_pot / kT)
        yH = 2.0 * np.sqrt(q) / (np.sqrt(q) + np.sqrt(4.0 + q))
        mu1 = u.mu1_0 * (1.0 + yH + u.xHe)
        return {"pp": u.Rgas * mu1 * rho * TT}

    def _load_t(self):
        return {"t": self._read_var(["lnrho"]).t}

    @property
    def time(self):
        """Snapshot time (code units)."""
        return self._get("t", self._load_t)

    @property
    def ux(self):
        return self._get("ux", self._load_uu)

    @property
    def uy(self):
        return self._get("uy", self._load_uu)

    @property
    def uz(self):
        return self._get("uz", self._load_uu)

    @property
    def bx(self):
        return self._get("bx", self._load_bb)

    @property
    def by(self):
        return self._get("by", self._load_bb)

    @property
    def bz(self):
        return self._get("bz", self._load_bb)

    @property
    def rho(self):
        return self._get("rho", self._load_rho)

    @property
    def TT(self):
        return self._get("TT", self._load_TT)

    @property
    def pp(self):
        """Pressure, including partial hydrogen ionisation (Saha equation)."""
        return self._get("pp", self._load_pp)


# Make the module itself callable: pencil_read(datapath) -> Reader(datapath)
class _CallableModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return Reader(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
