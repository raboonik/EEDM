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

    else:
        raise ValueError(f"Unsupported dataExt: {settings.dataExt}")


'''
import settings

# Conditional imports based on file type
if settings.dataExt == "sdf":
    try:
        import sdf
        __all__.append('sdf')
    except ImportError:
        raise ValueError("SDF module not found!")

elif settings.dataExt == "cfd":
    try:
        from .io.larexd import read_lare_cfd3d as cfd3d
        from .io.larexd import read_lare_cfd2d as cfd2d
        __all__.extend(['cfd3d', 'cfd2d'])
    except ImportError:
        raise ValueError("CFD module not found!")
'''