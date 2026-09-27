'''
    Add decorator functions to keep track of function calls.
'''


import functools
import time

from . import context

def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if context.rank == context.mainrank: print(f"[LOG] Calling {func.__name__}...\n")
        result = func(*args, **kwargs)
        if context.rank == context.mainrank: print(f"[LOG] Finished {func.__name__}\n")
        return result
    return wrapper

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        if context.rank == context.mainrank:
            h, rest = divmod(duration, 3600)
            m, sec  = divmod(rest, 60)
            print("[TIME] " + func.__name__ + " ran in " + (("%d h %d min " % (h, m)) if h else ("%d min " % m) if m else "") + "%.1f s\n" % sec)
        return result
    return wrapper

def memoize(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper
