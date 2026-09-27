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
import os

from .extensionLoader import get_io_modules
from . import larexd
from . import mancha
from . import pencil
from .. import system
from .. import utils
from .. import context

import settings

extensionReader = get_io_modules()

# Largest variation along an invariant axis (settings.invariantAxes), relative to the field's largest magnitude
invarianceTolerance = 1e-6

def select_snapshots():
    '''
        The snapshots selected by the settings (datapath, dataExt, dataInterval, and skip), in time
        order, without reading any data. Used by EEDM_reader and by Equation 9, so that both work on
        exactly the same snapshots. Returns (feed, extra, snapshots):
            feed     : the snapshot files (labels for the Pencil Code, whose snapshots have no extension)
            extra    : the paired Mancha "_extra.h5" files, or [] if there are none
            snapshots: the Pencil Code VAR indices, or None for the other codes
    '''
    extra, snapshots = [], None
    if settings.simCode == 'pencil':
        if settings.dataExt != 'var':
            raise ValueError('simCode = "pencil" needs dataExt = "var", not ' + repr(settings.dataExt) + ' (the snapshots are read from data/allprocs/VAR<N>).')
        snapshots = pencil.helper.open_run(settings.datapath).snapshots
        feed      = pencil.helper.get_feed(settings.datapath, snapshots)
    else:
        feed = system.fid(os.path.join(settings.datapath, '*.' + settings.dataExt))
    
    # Code specific helper call to clean the file list
    if 'mancha' in settings.simCode:
        feed, extra = mancha.helper.remove_extra_files(feed)
    
    if len(feed) == 0:
        what = "Pencil Code VAR" if settings.simCode == 'pencil' else "*." + settings.dataExt
        raise ValueError("No " + what + " snapshots found in " + settings.datapath + ". Check settings.datapath and settings.dataExt.")
    
    nFound      = len(feed)
    first, last = settings.dataInterval
    if last == -1: last = len(feed)
    select = slice(first, last + 1, settings.skip)
    
    feed = feed[select]
    if len(extra) > 0:         extra     = extra[select]      # keep the extra files paired with their snapshots
    if snapshots is not None:  snapshots = snapshots[select]
    
    if len(feed) == 0:
        raise ValueError("settings.dataInterval = " + str(settings.dataInterval) + " with skip = " + str(settings.skip) + " selects none of the " +
                         str(nFound) + " snapshots found in " + settings.datapath + " (indices 0 to " + str(nFound - 1) + ").")
    return feed, extra, snapshots

class EEDM_reader:
    def __init__(self):
        # Messages about the data found while setting up (printed after the run header)
        self.notes = []
        
        # Find the snapshots selected by the settings
        self.feed, self.extrah5, self.snapshots = select_snapshots()
        if settings.simCode == 'pencil':
            self.pencilRun = pencil.helper.open_run(settings.datapath)
        self.nt = len(self.feed)
        
        # The code dependent part
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
                raise ValueError('simCode = "lare" needs dataExt = "sdf" or "cfd", not ' + repr(settings.dataExt) + '.')
            
            del(dat)
            # The snapshot's full grid before cropping, for the run header (cell centres, and cell faces where the code has them)
            self.fullGrid  = [np.array(q, dtype=float) for q in (self.xc, self.yc, self.zc)]
            self.fullFaces = [np.array(q, dtype=float) for q in (self.xb, self.yb, self.zb)]
            self.xb, self.yb, self.zb, self.frameb, self.xc, self.yc, self.zc, self.framec  = \
                utils.crop.getCropIndecies3D_lare(self, settings.cropFramex, settings.cropFramey, settings.cropFramez)
        # MANCHA
        elif 'mancha' in settings.simCode:
            dat = extensionReader['h5'](self.feed[0], 'r')
            self.xb,self.yb,self.zb = None, None, None
            self.xc,self.yc,self.zc = mancha.helper.get_grid(dat)
            self.xb, self.yb, self.zb, self.frameb  = None, None, None, None
            # The snapshot's full grid before cropping, for the run header
            self.fullGrid  = [np.array(q, dtype=float) for q in (self.xc, self.yc, self.zc)]
            self.fullFaces = None
            self.xc, self.yc, self.zc, self.framec  =  \
                utils.crop.getCropIndecies3D_cc(self, settings.cropFramex, settings.cropFramey, settings.cropFramez)
            dat.close()
            
            # Check if there is a background
            self.bckgrndCond = False
            if settings.simCode != 'mancha_0':
                bckgrndF = system.fid(os.path.join(settings.datapath, 'background_plasma.h5'))
                if len(bckgrndF) == 0:
                    raise ValueError('simCode = "mancha" adds the background state from ' + os.path.join(settings.datapath, 'background_plasma.h5') +
                                     ', which does not exist. If the snapshots already hold the full fields (zero background), use simCode = "mancha_0".')
                else:
                    self.notes.append("Adding the background state from " + bckgrndF[0] + " to every snapshot.")
                    self.bckgrndCond = True
                    dat = extensionReader['h5'](bckgrndF[0])
                    self.rho0 = (np.array(dat['rho']).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    self.vx0  = (np.array(dat['vx' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    self.vy0  = (np.array(dat['vy' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    self.vz0  = (np.array(dat['vz' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    self.bx0  = (np.array(dat['bx' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    self.by0  = (np.array(dat['by' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    self.bz0  = (np.array(dat['bz' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    self.p0   = (np.array(dat['pe' ]).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                    # Keep the background in the working precision, so that it is added to each snapshot in that precision
                    self.rho0, self.vx0, self.vy0, self.vz0, self.bx0, self.by0, self.bz0, self.p0 = [np.asarray(q, dtype=context.dtype) for q in
                        (self.rho0, self.vx0, self.vy0, self.vz0, self.bx0, self.by0, self.bz0, self.p0)]
                    dat.close()
        # PENCIL
        elif settings.simCode == 'pencil':
            # Collocated grid (no staggering), ghost zones already removed
            self.xb, self.yb, self.zb, self.frameb = None, None, None, None
            self.xc, self.yc, self.zc = pencil.helper.get_grid(self.pencilRun, settings.dimensionality_switch)
            # The snapshot's full grid before cropping, for the run header
            self.fullGrid  = [np.array(q, dtype=float) for q in (self.xc, self.yc, self.zc)]
            self.fullFaces = None
            self.xc, self.yc, self.zc, self.framec = \
                utils.crop.getCropIndecies3D_cc(self, settings.cropFramex, settings.cropFramey, settings.cropFramez)
            
            warning = pencil.helper.check_gravity(self.pencilRun, settings.g, settings.dimensionality_switch, context.verticalAxis)
            if warning: self.notes.append(warning.strip())
        else:
            raise ValueError("settings.simCode = " + repr(settings.simCode) + ' is not supported. Use "lare", "mancha", "mancha_0", or "pencil", ' +
                             "or contact raboonik@gmail.com to add support for your code.")
        
        # Use the chosen precision for the grid (the staggered grid is None for collocated codes)
        self.xc, self.yc, self.zc = [np.asarray(q, dtype=context.dtype) for q in (self.xc, self.yc, self.zc)]
        if self.xb is not None:
            self.xb, self.yb, self.zb = [np.asarray(q, dtype=context.dtype) for q in (self.xb, self.yb, self.zb)]
        
        # 2D and 1D runs (settings.invariantAxes): every varying axis needs enough points for the cubic-spline derivatives,
        # while an invariant axis may have any number of points and is reduced to its first cell (checked in data())
        for a in "xyz":
            n = len(getattr(self, a + "c"))
            if a not in context.invariantAxes and n < 4:
                raise ValueError("The (cropped) grid has only " + str(n) + " point(s) along " + a + ", too few for the spline derivatives (at least 4). " +
                                 "If the simulation does not vary along " + a + " (a 2D or 1D run), add \"" + a + "\" to settings.invariantAxes.")
        for a in context.invariantAxes:
            setattr(self, a + "c", getattr(self, a + "c")[:1])
            if self.xb is not None: setattr(self, a + "b", getattr(self, a + "b")[:2])     # the two faces of the kept cell
        
        self.nx, self.ny, self.nz = np.array([self.xc.shape[0],self.yc.shape[0],self.zc.shape[0]])
        
        # xc, yc, zc (and xb, ...) are in the simulation frame, used for everything the user sees; ex, ey, ez are the same
        # cell centres in the EEDM frame (gravity along -z), used for the derivatives and the height z in phi = g z
        self.ex, self.ey, self.ez = [(self.xc, self.yc, self.zc)[k] for k in utils.orientation.PERM]
        
    def data(self, i):
        # LARE 3D
        if settings.simCode == 'lare':
            if settings.dataExt == 'sdf':
                dat = extensionReader['sdf'].read(self.feed[i])
                # Simulation time of the dumped step (Last_dump_time_requested is only the scheduled output time)
                self.time = float(dat.Header['time'])
                
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
                if i == 0: print("Reading the pressure from the _extra.h5 files\n")
                datExt = extensionReader['h5'](self.extrah5[i], 'r')
                p = (np.array(datExt['pe']).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                datExt.close()
                dat.close()
            else:
                p = (np.array(dat['pe']).T)[self.framec[0]:self.framec[1],self.framec[2]:self.framec[3],self.framec[4]:self.framec[5]]
                dat.close()
            
            if self.bckgrndCond:
                # Add the background in the working precision: an in-place += would keep the file's precision (e.g. float32),
                # losing the small perturbations that MANCHA stores separately from the background for exactly that reason
                rho, vx, vy, vz, bx, by, bz, p = [np.asarray(q, dtype=context.dtype) + q0 for q, q0 in
                    zip((rho, vx, vy, vz, bx, by, bz, p), (self.rho0, self.vx0, self.vy0, self.vz0, self.bx0, self.by0, self.bz0, self.p0))]
            
            # p   = (settings.gamma - 1) * en
            # del(en)
        # PENCIL
        elif settings.simCode == 'pencil':
            self.time, rho, vx, vy, vz, bx, by, bz, p = \
                pencil.helper.get_data(self.pencilRun, self.snapshots[i], self.framec, settings.dimensionality_switch)
        else: 
            # Code already handled
            pass
        
        fields = [rho, vx, vy, vz, bx, by, bz, p]
        
        # 2D and 1D runs: check that the data are uniform along each invariant axis, then keep its first cell
        for a in context.invariantAxes:
            axis = "xyz".index(a)
            for name, q in zip(("rho", "vx", "vy", "vz", "bx", "by", "bz", "p"), fields):
                q     = np.asarray(q)
                scale = np.max(np.abs(q))
                vary  = np.max(np.abs(q - np.take(q, [0], axis=axis)))
                if vary > invarianceTolerance * scale:
                    raise ValueError(name + " in " + os.path.basename(self.feed[i]) + " varies along " + a + " by up to %.3g of its largest magnitude, " % (vary / scale) +
                                     "but " + a + " is listed in settings.invariantAxes (allowed: %g). Check the data, or remove " % invarianceTolerance + a +
                                     " from invariantAxes to treat the run as varying along " + a + ".")
            fields = [np.take(q, [0], axis=axis) for q in fields]
        
        # Rotate into the EEDM frame (the identity when settings.verticalAxis = "z")
        rho, p            = utils.orientation.array_in(fields[0]), utils.orientation.array_in(fields[7])
        vx, vy, vz        = utils.orientation.vector_in(*fields[1:4])
        bx, by, bz        = utils.orientation.vector_in(*fields[4:7])
        
        # Hand over every field in the chosen precision, whatever the simulation code stored
        return tuple(np.asarray(q, dtype=context.dtype) for q in (rho, vx, vy, vz, bx, by, bz, p))


