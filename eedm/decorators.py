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
        if context.rank == context.mainrank: print(f"[TIME] {func.__name__} ran in {duration:.3f} seconds\n")
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

def cache(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper