'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: A class to read simulation data files. When initialized, 
                 it first reads all the metadata and stores it in memory.
                 A further call to "data" is then needed to read the MHD 
                 data depending on the simulation code and data format.
'''
import numpy as np

from .extensionLoader import get_io_modules
from . import larexd
from .. import system
from .. import utils

import settings

extensionReader = get_io_modules()

class EEDM_reader:
    def __init__(self):
        
        # The code independent part of the reader
        self.feed = system.fid(settings.datapath + '*.' + settings.dataExt)
        self.nt   = len(self.feed)
        
        if settings.dataInterval[1] == -1: settings.dataInterval[1] = self.nt
        
        self.feed = self.feed[settings.dataInterval[0]:settings.dataInterval[1]+1:settings.skip]
        self.nt   = len(self.feed)
        
        if self.nt == 0: raise ValueError("No snapshots found in " + settings.datapath)
        
        # The code dependent part
        if settings.simCode == 'lare':
            if settings.dim == 3:
                if settings.dataExt == 'sdf':
                    dat = extensionReader['sdf'].read(self.feed[0])
                    self.xb,self.yb,self.zb = dat.__dict__["Grid_Grid"].data
                    self.xc,self.yc,self.zc = dat.__dict__["Grid_Grid_mid"].data
                elif settings.dataExt == 'cfd':
                    dat = extensionReader['cfd3d'].read_lare_cfd3d(self.feed[0])
                    self.xb,self.yb,self.zb = dat["x"],dat["y"],dat["z"]
                    self.xc,self.yc,self.zc = dat["xc"], dat["yc"], dat["zc"]
                else:
                    raise ValueError("Unknown data format: " + settings.dataExt + ". Contact raboonik@gmail.com to add support for your data format.")
                
                del(dat)
                self.xb, self.yb, self.zb, self.xc, self.yc, self.zc, self.frameb, self.framec  = \
                    utils.crop.getCropIndecies3D(self, settings.cropFramex, settings.cropFramey, settings.cropFramez)
                self.nx, self.ny, self.nz = np.array([self.xc.shape[0],self.yc.shape[0],self.zc.shape[0]])
                
            elif settings.dim == 2:
                raise ValueError("2D support is under development but not implemented yet.")
        else:
            raise ValueError("Unknown simulation code: " + settings.simCode + ". Contact raboonik@gmail.com to add support for your code.")
        
    def data(self, i):
        if settings.dim == 3:
            if settings.simCode == 'lare':
                if settings.dataExt == 'sdf':
                    dat = extensionReader['sdf'].read(self.feed[i])
                    self.time = dat.__dict__['Last_dump_time_requested'].data
                    
                    rho = dat.__dict__['Fluid_Rho'        ].data[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    en  = dat.__dict__['Fluid_Energy'     ].data[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    vx  = dat.__dict__['Velocity_Vx'      ].data[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    vy  = dat.__dict__['Velocity_Vy'      ].data[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    vz  = dat.__dict__['Velocity_Vz'      ].data[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    bx  = dat.__dict__['Magnetic_Field_Bx'].data[self.frameb[0]:self.frameb[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    by  = dat.__dict__['Magnetic_Field_By'].data[self.framec[0]:self.framec[1],self.frameb[2]:self.frameb[3],self.framec[4]:self.framec[5]]
                    bz  = dat.__dict__['Magnetic_Field_Bz'].data[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.frameb[4]:self.frameb[5]]
                    
                elif settings.dataExt == 'cfd':
                    dat = extensionReader['cfd3d'].read_lare_cfd3d(self.feed[i])
                    self.time = dat["time"]
                    
                    rho = (np.transpose(dat["rho"]   )[1:,1:,1:])[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    en  = (np.transpose(dat["energy"])[1:,1:,1:])[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    vx  =  np.transpose(dat["vx"]               )[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    vy  =  np.transpose(dat["vy"]               )[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    vz  =  np.transpose(dat["vz"]               )[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    bx  = (np.transpose(dat["bx"]    )[:,1:,1:] )[self.frameb[0]:self.frameb[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    by  = (np.transpose(dat["by"]    )[1:,:,1:] )[self.framec[0]:self.framec[1],self.frameb[2]:self.frameb[3],self.framec[4]:self.framec[5]]
                    bz  = (np.transpose(dat["bz"]    )[1:,1:,:] )[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.frameb[4]:self.frameb[5]]
                else:
                    # Extension already handled
                    pass
                
                p   = (settings.gamma - 1) * rho * en
                del(en)
                
                # Compute v and B at cell centers
                vx = larexd.lare.getCellVals3D(vx,"v" )
                vy = larexd.lare.getCellVals3D(vy,"v" )
                vz = larexd.lare.getCellVals3D(vz,"v" )
                bx = larexd.lare.getCellVals3D(bx,"bx")
                by = larexd.lare.getCellVals3D(by,"by")
                bz = larexd.lare.getCellVals3D(bz,"bz")
            else: 
                # Code already handled
                pass
            
            del(dat)
        else: 
            # 2D already handled
            pass
        
        return rho, vx, vy, vz, bx, by, bz, p


