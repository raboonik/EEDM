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
from . import mancha
from .. import system
from .. import utils
from .. import context

import settings

extensionReader = get_io_modules()

class EEDM_reader:
    def __init__(self):
        
        # Find the files
        self.feed = system.fid(settings.datapath + '*.' + settings.dataExt)
        
        
        # Code specific helper call to clean the file list
        if 'mancha' in settings.simCode:
            self.feed, self.extrah5 = mancha.helper.remove_extra_files(self.feed)
        else:
            pass
        
        self.nt   = len(self.feed)
        
        if settings.dataInterval[1] == -1: settings.dataInterval[1] = self.nt
        
        self.feed = self.feed[settings.dataInterval[0]:settings.dataInterval[1]+1:settings.skip]
        self.nt   = len(self.feed)
        
        if self.nt == 0: raise ValueError("No snapshots found in " + settings.datapath)
        
        # The code dependent part
        if settings.dim == 3:
            # LARE 3D
            if settings.simCode == 'lare':
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
                self.xb, self.yb, self.zb, self.frameb, self.xc, self.yc, self.zc, self.framec  = \
                    utils.crop.getCropIndecies3D_lare(self, settings.cropFramex, settings.cropFramey, settings.cropFramez)
            # MANCHA
            elif 'mancha' in settings.simCode:
                dat = extensionReader['h5'](self.feed[0], 'r')
                self.xb,self.yb,self.zb = None, None, None
                self.xc,self.yc,self.zc = mancha.helper.get_grid(dat)
                self.xb, self.yb, self.zb, self.frameb  = None, None, None, None
                self.xc, self.yc, self.zc, self.framec  =  \
                    utils.crop.getCropIndecies3D_cc(self, settings.cropFramex, settings.cropFramey, settings.cropFramez)
                dat.close()
                
                # Check if there is a background
                self.bckgrndCond = False
                if settings.simCode != 'mancha_0':
                    bckgrndF = system.fid(settings.datapath + 'background_plasma.h5')
                    if len(bckgrndF) == 0:
                        raise ValueError("No background file found in " + settings.datapath + \
                                        ". If there the background is zero, use 'simCode = mancha_0' instead.")
                    elif len(bckgrndF) > 1:
                        raise ValueError("Multiple background files found in " + settings.datapath + \
                                        ". Please remove all but one and try again.")
                    else:
                        if context.rank == context.mainrank: print("Reading the background plasma: " + bckgrndF[0]+"\n")
                        self.bckgrndCond = True
                        dat = extensionReader['h5'](bckgrndF[0])
                        self.rho0 = (np.array(dat['rho']).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                        self.vx0  = (np.array(dat['vx' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                        self.vy0  = (np.array(dat['vy' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                        self.vz0  = (np.array(dat['vz' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                        self.bx0  = (np.array(dat['bx' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                        self.by0  = (np.array(dat['by' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                        self.bz0  = (np.array(dat['bz' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                        self.p0   = (np.array(dat['bz' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                        dat.close()
            else:
                raise ValueError("Unknown simulation code: " + settings.simCode + ". Contact raboonik@gmail.com to add support for your code.")
            
            self.nx, self.ny, self.nz = np.array([self.xc.shape[0],self.yc.shape[0],self.zc.shape[0]])
            
        elif settings.dim == 2:
                raise ValueError("2D support is under development but not implemented yet. Contact raboonik@gmail.com.")
        
        elif settings.dim == 1:
                raise ValueError("1D support is under development but not implemented yet. Contact raboonik@gmail.com.")
        
        else:
            raise ValueError("Invalid dimensionality!")
        
    def data(self, i):
        if settings.dim == 3:
            # LARE 3D
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
                vx = larexd.helper.getCellVals3D(vx,"v" )
                vy = larexd.helper.getCellVals3D(vy,"v" )
                vz = larexd.helper.getCellVals3D(vz,"v" )
                bx = larexd.helper.getCellVals3D(bx,"bx")
                by = larexd.helper.getCellVals3D(by,"by")
                bz = larexd.helper.getCellVals3D(bz,"bz")
                
                del(dat)
            # MANCHA
            elif 'mancha' in settings.simCode:
                dat = extensionReader['h5'](self.feed[i], 'r')
                self.time = dat.attrs['time'][0]
                
                rho = (np.array(dat['rho']).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                vx  = (np.array(dat['vx' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                vy  = (np.array(dat['vy' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                vz  = (np.array(dat['vz' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                bx  = (np.array(dat['bx' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                by  = (np.array(dat['by' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                bz  = (np.array(dat['bz' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                
                if len(self.extrah5) > 0:
                    if i == 0: print("Reading pressure form extra h5 files...\n")
                    datExt = extensionReader['h5'](self.extrah5[i], 'r')
                    p = (np.array(datExt['pe']).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    datExt.close()
                else:
                    p = (np.array(dat['pe']).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    dat.close()
                
                if self.bckgrndCond:
                    rho += self.rho0
                    vx  += self.vx0
                    vy  += self.vy0
                    vz  += self.vz0
                    bx  += self.bx0
                    by  += self.by0
                    bz  += self.bz0
                    p   += self.p0
                
                # p   = (settings.gamma - 1) * en
                # del(en)
            else: 
                # Code already handled
                pass
        else: 
            # other dimensionalities already handled
            pass
        
        return rho, vx, vy, vz, bx, by, bz, p


