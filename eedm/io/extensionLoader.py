import settings

def get_io_modules():
    if settings.dataExt == "sdf":
        try:
            import sdf
            return {"sdf": sdf}
        except ImportError as e:
            raise ImportError("Reading .sdf snapshots needs the \"sdf\" Python module from the LARE/EPOCH SDF tools, which could not be imported (" +
                              str(e) + "). Build it and add it to your PYTHONPATH.") from e

    elif settings.dataExt == "cfd":
        try:
            from .larexd import read_lare_cfd3d
            from .larexd import read_lare_cfd2d
            return {"cfd3d": read_lare_cfd3d, "cfd2d": read_lare_cfd2d}
        except ImportError as e:
            raise ImportError("Could not import EEDM's LARE .cfd reader (" + str(e) + ").") from e
        
    elif settings.dataExt == "h5":
        try:
            import h5py
            return {"h5": h5py.File}
        except ImportError as e:
            raise ImportError("Reading .h5 snapshots needs h5py, which could not be imported (" + str(e) + "). Install it with: pip install h5py") from e

    elif settings.dataExt == "var":
        try:
            import pencil  # Pencil Code's python package, used by pencil_read
            from .pencil import pencil_read
            return {"var": pencil_read}
        except ImportError as e:
            raise ImportError("Reading Pencil Code snapshots needs the Pencil Code Python package, which could not be imported (" + str(e) +
                              "). Add pencil-code/python to your PYTHONPATH.") from e

    else:
        raise ValueError("settings.dataExt = " + repr(settings.dataExt) + ' is not supported. Use "sdf" or "cfd" (LARE), "h5" (MANCHA), or "var" (Pencil Code).')
