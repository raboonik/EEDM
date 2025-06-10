from libs import prerequisites as pre

'''
    Reader of simulation data files.
    
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    A class to read simulation data files. When initialized, it first reads all the metadata and stores it in memory.
    A further call to "data" is then needed to read the MHD data depending on the simulation code and data format.
'''

class EEDM_reader:
    def __init__(self):        
        self.feed = pre.fid(pre.datapath + '*.' + pre.dataExt)
        self.nt   = len(self.feed)
        
        if pre.dataInterval[1] == -1: pre.dataInterval[1] = self.nt
        
        self.feed = self.feed[pre.dataInterval[0]:pre.dataInterval[1]+1:pre.skip]
        self.nt   = len(self.feed)
        
        if self.nt == 0: raise ValueError("No snapshots found in " + pre.datapath)
        
        if pre.simCode == 'lare':
            if pre.dim == 3:
                if pre.dataExt == 'sdf':
                    dat = pre.sdf.read(self.feed[0])
                    self.xb,self.yb,self.zb = dat.__dict__["Grid_Grid"].data
                    self.xc,self.yc,self.zc = dat.__dict__["Grid_Grid_mid"].data
                elif pre.dataExt == 'cfd':
                    dat = pre.cfd3d.read_lare_cfd3d(self.feed[0])
                    self.xb,self.yb,self.zb = dat["x"],dat["y"],dat["z"]
                    self.xc,self.yc,self.zc = dat["xc"], dat["yc"], dat["zc"]
                else:
                    raise ValueError("Unknown data format: " + pre.dataExt + ". Contact raboonik@gmail.com to add support for your data format.")
                
                del(dat)
                self.xb, self.yb, self.zb, self.xc, self.yc, self.zc, self.frameb, self.framec  = pre.getCropIndecies3D(self, pre.cropFramex, pre.cropFramey, pre.cropFramez)
                self.nx, self.ny, self.nz = pre.np.array([self.xc.shape[0],self.yc.shape[0],self.zc.shape[0]])
                
            elif pre.dim == 2:
                raise ValueError("2D support is under development but not implemented yet.")
        else:
            raise ValueError("Unknown simulation code: " + pre.simCode + ". Contact raboonik@gmail.com to add support for your code.")
        
    def data(self, i):
        if pre.dim == 3:
            if pre.simCode == 'lare':
                if pre.dataExt == 'sdf':
                    dat = pre.sdf.read(self.feed[i])
                    self.time = dat.__dict__['Last_dump_time_requested'].data
                    
                    rho = dat.__dict__['Fluid_Rho'        ].data[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    en  = dat.__dict__['Fluid_Energy'     ].data[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    vx  = dat.__dict__['Velocity_Vx'      ].data[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    vy  = dat.__dict__['Velocity_Vy'      ].data[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    vz  = dat.__dict__['Velocity_Vz'      ].data[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    bx  = dat.__dict__['Magnetic_Field_Bx'].data[self.frameb[0]:self.frameb[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    by  = dat.__dict__['Magnetic_Field_By'].data[self.framec[0]:self.framec[1],self.frameb[2]:self.frameb[3],self.framec[4]:self.framec[5]]
                    bz  = dat.__dict__['Magnetic_Field_Bz'].data[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.frameb[4]:self.frameb[5]]
                    
                elif pre.dataExt == 'cfd':
                    dat = pre.cfd3d.read_lare_cfd3d(self.feed[i])
                    self.time = dat["time"]
                    
                    rho = (pre.np.transpose(dat["rho"]   )[1:,1:,1:])[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    en  = (pre.np.transpose(dat["energy"])[1:,1:,1:])[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    vx  =  pre.np.transpose(dat["vx"])[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    vy  =  pre.np.transpose(dat["vy"])[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    vz  =  pre.np.transpose(dat["vz"])[self.frameb[0]:self.frameb[1],self.frameb[2]:self.frameb[3],self.frameb[4]:self.frameb[5]]
                    bx  = (pre.np.transpose(dat["bx"])[:,1:,1:])[self.frameb[0]:self.frameb[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    by  = (pre.np.transpose(dat["by"])[1:,:,1:])[self.framec[0]:self.framec[1],self.frameb[2]:self.frameb[3],self.framec[4]:self.framec[5]]
                    bz  = (pre.np.transpose(dat["bz"])[1:,1:,:])[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.frameb[4]:self.frameb[5]]
                else:
                    # Extension already handled
                    pass
                
                p   = (pre.gamma - 1) * rho * en
                del(en)
                
                # Compute v and B at cell centers
                vx = pre.lare3dGetCellVals(vx,"v")
                vy = pre.lare3dGetCellVals(vy,"v")
                vz = pre.lare3dGetCellVals(vz,"v")
                bx = pre.lare3dGetCellVals(bx,"bx")
                by = pre.lare3dGetCellVals(by,"by")
                bz = pre.lare3dGetCellVals(bz,"bz")
            else: 
                # Code already handled
                pass
            
            del(dat)
        else: 
            # 2D already handled
            pass
        
        return rho, p, vx, vy, vz, bx, by, bz