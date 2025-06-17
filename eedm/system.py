'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: Script to include all the libraries required for the EEDM code
'''

import subprocess
import glob
import os

def system_call(command):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if len(err) > 5:
        return 0
    else:
        return 1

def fid(path):
    return sorted(glob.glob(path))

def add_slash(path):
    if path[-1] != '/': path = path + '/'
    return path

def bash(cmd): return os.system(cmd)
def  cd(path):  os.chdir(path)