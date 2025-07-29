import settings

def get_io_modules():
    if settings.dataExt == "sdf":
        try:
            import sdf
            return {"sdf": sdf}
        except ImportError:
            raise ValueError("SDF module not found!")

    elif settings.dataExt == "cfd":
        try:
            from .larexd import read_lare_cfd3d
            from .larexd import read_lare_cfd2d
            return {"cfd3d": read_lare_cfd3d, "cfd2d": read_lare_cfd2d}
        except ImportError:
            raise ValueError("CFD module not found!")
        
    elif settings.dataExt == "h5":
        try:
            import h5py
            return {"h5": h5py.File}
        except ImportError:
            raise ValueError("h5py module not found!")

    else:
        raise ValueError(f"Unsupported dataExt: {settings.dataExt}")