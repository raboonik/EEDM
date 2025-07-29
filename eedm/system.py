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

def system_call(command,quiet=False):
    if quiet:
        p = subprocess.run([command], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        p = subprocess.run([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if p.returncode == 0:
        return 0
    else:
        return 1

def create_dir(dir):
    err = system_call("cd " + dir, quiet=True)
    if err != 0:
        print("Creating the directory: "+dir+"\n")
        err = system_call("mkdir "+dir,quiet=True)
    else:
        print("Directory already exists: "+dir+"\n")
    
    return err

def fid(path):
    return sorted(glob.glob(path))

def add_slash(path):
    if path[-1] != '/': path = path + '/'
    return path

def bash(cmd): return os.system(cmd)
def  cd(path):  os.chdir(path)