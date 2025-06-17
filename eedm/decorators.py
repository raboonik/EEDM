'''
    Add decorator functions to keep track of function calls.
'''


import functools
import time

def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[LOG] Calling {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"[LOG] Finished {func.__name__}")
        return result
    return wrapper

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"[TIME] {func.__name__} ran in {duration:.3f} seconds")
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