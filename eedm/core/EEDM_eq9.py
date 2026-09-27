'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: Script to compute Equation 9 of Paper III (linked above)
'''

import numpy as np
import os
import contextlib
import h5py

try:
    from scipy.integrate import cumulative_trapezoid as timeIntegral
except ImportError:
    from scipy.integrate import cumtrapz as timeIntegral


from .. import context as ct
from .. import io
from .. import utils
from .. import system
from .. import APP
from .. import decorators

import settings as set

@decorators.memoize
def run(dirDict):
    
    # Use exactly the Equation 6 outputs of the snapshots selected by the current settings, in time order,
    # so that leftovers from earlier runs in the same folder are never mixed in
    stems = [os.path.splitext(os.path.basename(f))[0] for f in io.reader.select_snapshots()[0]]
    
    def selected_files(key, prefix):
        files   = [os.path.join(dirDict[key], prefix + stem + '.h5') for stem in stems]
        missing = [f for f in files if not os.path.isfile(f)]
        if len(missing) > 0:
            raise ValueError("Equation 9 needs the " + prefix + "*.h5 file of every snapshot selected in settings.py, but " + str(len(missing)) + " of " +
                             str(len(files)) + " are missing from " + dirDict[key] + " (e.g. " + os.path.basename(missing[0]) + "). Compute Equation 6 " +
                             'with the same settings first (computationSwitch = "eq6" or "both").')
        return files
    
    # The eigenenergy time derivatives, and if requested, the energy, characteristic speed, and divB/entropy (Extras) data
    fid1      = selected_files("eq6", "EigenenergyDDT_")
    filenames = [os.path.basename(f) for f in fid1]
    nt        = len(fid1)
    fidEn     = selected_files("energy", "Etot_")   if ct.EnCond               else []
    fidSp     = selected_files("speed" , "Speed_")  if ct.SpCond               else []
    fidEx     = selected_files("extra" , "Extras_") if ct.DbCond or ct.PkCond  else []
    
    unused = len(system.fid(os.path.join(dirDict["eq6"], 'EigenenergyDDT_*.h5'))) - nt
    if unused > 0 and ct.rank == ct.mainrank:
        print("Note: ignoring " + str(unused) + " Equation 6 file(s) in " + dirDict["eq6"] + " that the current settings do not select.\n")
    
    dat      = h5py.File(os.path.join(dirDict['parent'], 'grid_params.h5'),'r') 
    g        = dat.attrs["g"]
    GrCond   = dat.attrs["gCond"]
    xb       = np.array(dat["xc"])
    yb       = np.array(dat["yc"])
    zb       = np.array(dat["zc"])
    dat.close()
    
    ct.comm.barrier()
    
    # Although here we're only considering the full physical domain, we add the my prefix to avoid having to repeat essentially the same lines of code
    # with tiny tweaks to differentiate 1D and 4D. We will modify these quantities if dim = "4D-parallelize"
    # The slab indices along the plane's normal (checked again here, so that an Equation 9 only run is checked too)
    slabIdx, _, notes = utils.get_energy_sliced.resolve_slicing_points(set.slicingPlane, set.slicingPnts, getattr(set, "slicingPntsType", None), xb, yb, zb)
    if ct.rank == ct.mainrank and not ct.eq6Cond:
        for note in notes: print(note)
    
    nx,ny,nz,axis1,axis2,axis3,slicingPnts,slicingPntsL, mylenq, mynq1, mynq2, mynq1i, mynq1f, mynq2i, mynq2f, mynq3i, mynq3f, outfilename1, outfilename2 = \
            utils.get_energy_sliced.get_slice_2d(set.slicingPlane, slabIdx, xb,yb,zb, dirDict["eq9"])
    slicingPnts = np.array(slicingPnts, dtype=int)
    
    # Note that here mylenq,mynq1,mynq2 = lenq,nq1,nq2
    lenq,nq1,nq2 = mylenq,mynq1,mynq2
    axes_limits  = [lenq,nq1,nq2,nt]
    
    if ct.rank == ct.mainrank:
        normal = "xyz".strip(set.slicingPlane)
        print("Equation 9: time integrals over " + str(nt) + " snapshot(s) on " + str(lenq) + " " + set.slicingPlane + "-slab(s), at " +
              ", ".join("%s = %g (index %d)" % (normal, float(loc), int(idx)) for loc, idx in zip(slicingPntsL, slicingPnts)), flush=True)
    
    # Parallelize the first 3 axes, since we want the entire time axis to avoid having to call reshape_array_ND which is costly
    parallel_axes = [0,1,2]
    domDecompND   = APP.domainDecomposeND(ct.size, axes_limits, parallel_axes)
    slPnts        = domDecompND.slq[0][ct.rank]
    elPnts        = domDecompND.elq[0][ct.rank]
    slq1          = domDecompND.slq[1][ct.rank]
    elq1          = domDecompND.elq[1][ct.rank]
    slq2          = domDecompND.slq[2][ct.rank]
    elq2          = domDecompND.elq[2][ct.rank]
    
    # AutoParallelizePy transfers arrays as MPI.DOUBLE: send float64, then return to the chosen precision
    def gather(subArr):
        out = APP.mpi.gather_array_ND(ct.comm, ct.rank, ct.mainrank, domDecompND, np.asarray(subArr, dtype=np.float64))
        return None if out is None else out.astype(ct.dtype, copy=False)
    
    if ct.verbose and ct.rank == ct.mainrank:
        print("  domain decomposition over " + str(ct.size) + " rank(s): axes_limits = " + str(axes_limits) + ", parallel_axes = " + str(parallel_axes))
        print("  first indices per rank (slq):\n" + str(domDecompND.slq))
        print("  end indices per rank (elq):\n" + str(domDecompND.elq), flush=True)
    
    # Get the sub-sub-spatial slicing indeces for each proc
    if   set.slicingPlane == "xy":
        _,_,_,_,_,_,mynq0,_, mylenq, mynq1, mynq2, mynq1i,   \
        mynq1f, mynq2i, mynq2f, mynq3i, mynq3f, _,_ = \
        utils.get_energy_sliced.get_slice_2d(set.slicingPlane, slicingPnts[slPnts:elPnts], xb[slq1:elq1],yb[slq2:elq2],zb, dirDict["eq9"], slq1,elq1,slq2,elq2)
    elif set.slicingPlane == "xz":
        _,_,_,_,_,_,mynq0,_, mylenq, mynq1, mynq2, mynq1i,   \
        mynq1f, mynq2i, mynq2f, mynq3i, mynq3f, _,_ = \
        utils.get_energy_sliced.get_slice_2d(set.slicingPlane, slicingPnts[slPnts:elPnts], xb[slq1:elq1],yb,zb[slq2:elq2], dirDict["eq9"], slq1,elq1,slq2,elq2)
    elif set.slicingPlane == "yz":
        _,_,_,_,_,_,mynq0,_, mylenq, mynq1, mynq2, mynq1i,   \
        mynq1f, mynq2i, mynq2f, mynq3i, mynq3f, _,_ = \
        utils.get_energy_sliced.get_slice_2d(set.slicingPlane, slicingPnts[slPnts:elPnts], xb,yb[slq1:elq1],zb[slq2:elq2], dirDict["eq9"], slq1,elq1,slq2,elq2)
    
    myshape = domDecompND.mynq[:,ct.rank]
    
    if ct.verbose:
        print("  [rank %d] sub-slab shape %s; index ranges %s:%s, %s:%s, %s:%s" % (ct.rank, [int(v) for v in myshape], [int(v) for v in mynq1i], [int(v) for v in mynq1f],
              [int(v) for v in mynq2i], [int(v) for v in mynq2f], [int(v) for v in mynq3i], [int(v) for v in mynq3f]), flush=True)
    
    # The eigenenergy outputs of the chosen decomposition (the g output only with gravity), plus the divB error term of Paper II
    # Outputs along invariant axes (2D and 1D runs) are [0] placeholders: they are written as such to the slabs, not integrated
    empty = utils.decomposition.placeholders(gravity=GrCond)
    names = [name for name in utils.decomposition.outputs(gravity=GrCond) if name not in empty]
    if ct.ErCond: names.append('m1_err')
    
    # Equation 6 must have been computed with the same decomposition
    with h5py.File(fid1[0], 'r') as dat:
        missing = [name for name in names if 'eq6_' + name not in dat]
    if len(missing) > 0:
        with h5py.File(os.path.join(dirDict['parent'], 'grid_params.h5'), 'r') as dat:
            used = dat.attrs.get('decomposition', 'unknown')
        raise ValueError("The Equation 6 files in " + dirDict["eq6"] + " were computed with decomposition = " + repr(used) + " and have no " +
                         ", ".join('eq6_' + name for name in missing) + ". Compute Equation 6 again with decomposition = " + repr(ct.decomposition) +
                         ", or run Equation 9 with decomposition = " + repr(used) + ".")
    
    myDDTen = {name: np.zeros(myshape, dtype=ct.dtype) for name in names}
    
    if ct.SpCond:
        myax         = np.zeros(myshape, dtype=ct.dtype)
        myay         = np.zeros(myshape, dtype=ct.dtype)
        myaz         = np.zeros(myshape, dtype=ct.dtype)
        mycsx        = np.zeros(myshape, dtype=ct.dtype)
        mycsy        = np.zeros(myshape, dtype=ct.dtype)
        mycsz        = np.zeros(myshape, dtype=ct.dtype)
        mycfx        = np.zeros(myshape, dtype=ct.dtype)
        mycfy        = np.zeros(myshape, dtype=ct.dtype)
        mycfz        = np.zeros(myshape, dtype=ct.dtype)
        mycs         = np.zeros(myshape, dtype=ct.dtype)
        
    if ct.EnCond:
        myEtot       = np.zeros(myshape, dtype=ct.dtype)
    
    # Since t is not parallelized, we can simply read time from HDF times for each ct.rank individually
    time        = np.zeros(nt)
    
    if ct.DbCond: 
        maxDivB     = np.zeros(nt, dtype=ct.dtype)
        meanDivB    = np.zeros(nt, dtype=ct.dtype)
        stdDivB     = np.zeros(nt, dtype=ct.dtype)
    if ct.PkCond:
        maxEntropy  = np.zeros(nt, dtype=ct.dtype)
        meanEntropy = np.zeros(nt, dtype=ct.dtype)
        stdEntropy  = np.zeros(nt, dtype=ct.dtype)
    
    # A slab read from the 3D data has shape (n1, 1, n2) (or similar): drop only the axis normal to the slab. np.squeeze would
    # also drop an in-plane axis that happens to be 1 point wide on this rank, and the result would no longer fit
    def in_plane(a):
        return np.reshape(a, (int(myshape[1]), int(myshape[2])))
    
    for i in range(nt):
        # Read through Python file objects (as before, so HDF5's own file locking is not involved), and close every
        # HDF5 file and its file object at the end of the iteration, even if an error occurs
        with contextlib.ExitStack() as files:
            def open_h5(path):
                return files.enter_context(h5py.File(files.enter_context(open(path, 'rb')), 'r'))
            
            filename = filenames[i]
            if ct.rank == ct.mainrank: print("  reading %s (%d/%d)" % (filename, i + 1, nt), flush=True)
            dat   = open_h5(fid1[i])
            
            time[i] = float(dat.attrs['time'])
            
            if ct.SpCond: datSp = open_h5(fidSp[i])
            
            if ct.EnCond: datEn = open_h5(fidEn[i])
            
            # The divB and entropy statistics are stored in the Extras files by Eq6
            if ct.DbCond or ct.PkCond: datEx = open_h5(fidEx[i])
            
            if ct.DbCond: 
                maxDivB[i]     = float(datEx.attrs['maxAbsDivB'    ])
                meanDivB[i]    = float(datEx.attrs['meanAbsDivB'   ])
                stdDivB[i]     = float(datEx.attrs['stdAbsDivB'    ])
            
            if ct.PkCond:
                maxEntropy[i]  = float(datEx.attrs['maxAbsEntropy' ])
                meanEntropy[i] = float(datEx.attrs['meanAbsEntropy'])
                stdEntropy[i]  = float(datEx.attrs['stdAbsEntropy' ])
            
            for ipnt in range(mylenq):
                for name in names:
                    myDDTen[name][ipnt,:,:,i] = in_plane(dat["eq6_" + name][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
                
                if ct.SpCond:
                    myax[  ipnt,:,:,i]      = in_plane(   datSp["ax"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
                    myay[  ipnt,:,:,i]      = in_plane(   datSp["ay"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
                    myaz[  ipnt,:,:,i]      = in_plane(   datSp["az"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
                    mycsx[ ipnt,:,:,i]      = in_plane(  datSp["csx"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
                    mycsy[ ipnt,:,:,i]      = in_plane(  datSp["csy"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
                    mycsz[ ipnt,:,:,i]      = in_plane(  datSp["csz"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
                    mycfx[ ipnt,:,:,i]      = in_plane(  datSp["cfx"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]]) 
                    mycfy[ ipnt,:,:,i]      = in_plane(  datSp["cfy"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
                    mycfz[ ipnt,:,:,i]      = in_plane(  datSp["cfz"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
                    
                    mycs[  ipnt,:,:,i]      = np.sqrt(mycfz[ipnt,:,:,i]**2 + mycsz[ipnt,:,:,i]**2 - (myax[ipnt,:,:,i]**2 + myay[ipnt,:,:,i]**2 + myaz[ipnt,:,:,i]**2))
                
                if ct.EnCond:
                    myEtot[ipnt,:,:,i]      = in_plane(  datEn["Kin"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]] +
                                                           datEn["Mag"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]] +
                                                           datEn["Int"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
                    
                    if GrCond: 
                        myEtot[ipnt,:,:,i]      = in_plane(datEn["Grv"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]]) + \
                                                             myEtot[ipnt,:,:,i]
    
    # Gather everything on ct.mainrank to export
    if ct.rank == ct.mainrank:
        print("  writing the slab file(s): " + ", ".join(os.path.basename(f) for f in outfilename1[:lenq]), flush=True)
    
    if ct.SpCond:
        ax  = gather(myax)
        ay  = gather(myay)
        az  = gather(myaz)
    
    if ct.rank == ct.mainrank:
        timesteps = (np.roll(time,-1,axis=0) - time)[0:-1]
        for ipnt in range(lenq):
            hdf = h5py.File(outfilename1[ipnt], "w")
            hdf.attrs['slicingPnt']   = slicingPnts[ipnt]
            hdf.attrs['slicingPntL']  = slicingPntsL[ipnt]
            hdf.attrs['slicingPlane'] = set.slicingPlane
            hdf.attrs['nt']           = nt
            hdf.attrs['nx']           = nx
            hdf.attrs['ny']           = ny
            hdf.attrs['nz']           = nz
            hdf.attrs['nq1']          = nq1
            hdf.attrs['nq2']          = nq2
            hdf.attrs['axis1']        = axis1
            hdf.attrs['axis2']        = axis2
            hdf.attrs['axis3']        = axis3
            hdf.attrs['g']            = g
            hdf.create_dataset('xc'          ,   data=np.array(xb,              dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdf.create_dataset('yc'          ,   data=np.array(yb,              dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdf.create_dataset('zc'          ,   data=np.array(zb,              dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdf.create_dataset('time'        ,   data=np.array(time,            dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('timesteps'   ,   data=np.array(timesteps,       dtype='float64'), compression='gzip', compression_opts=9)
            if ct.SpCond:
                hdf.create_dataset('ax'          ,   data=np.array(ax[ipnt,:,:,:] , dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('ay'          ,   data=np.array(ay[ipnt,:,:,:] , dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('az'          ,   data=np.array(az[ipnt,:,:,:] , dtype=ct.dtype), compression='gzip', compression_opts=9)
            if ct.DbCond:
                hdf.create_dataset('maxDivB'     ,   data=np.array(maxDivB,         dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('meanDivB'    ,   data=np.array(meanDivB,        dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('stdDivB'     ,   data=np.array(stdDivB,         dtype=ct.dtype), compression='gzip', compression_opts=9)
            if ct.PkCond:
                hdf.create_dataset('maxEntropy'  ,   data=np.array(maxEntropy,      dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('meanEntropy' ,   data=np.array(meanEntropy,     dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('stdEntropy'  ,   data=np.array(stdEntropy,      dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdf.close()
    
    if ct.SpCond:
        del(myax,
            myay,
            myaz,
            ax  ,
            ay  ,
            az)
        
        csx = gather(mycsx)
        csy = gather(mycsy)
        csz = gather(mycsz)
        cfx = gather(mycfx)
        cfy = gather(mycfy)
        cfz = gather(mycfz)
        cs  = gather(mycs)
        
        if ct.rank == ct.mainrank:
            for ipnt in range(lenq):
                hdf = h5py.File(outfilename1[ipnt], "a")
                hdf.create_dataset('csx',         data=np.array(csx[ipnt,:,:,:], dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('csy',         data=np.array(csy[ipnt,:,:,:], dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('csz',         data=np.array(csz[ipnt,:,:,:], dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('cfx',         data=np.array(cfx[ipnt,:,:,:], dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('cfy',         data=np.array(cfy[ipnt,:,:,:], dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('cfz',         data=np.array(cfz[ipnt,:,:,:], dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.create_dataset('cs' ,         data=np.array(cs[ipnt,:,:,:] , dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.close()
        
        del(mycsx,
            mycsy,
            mycsz,
            mycfx,
            mycfy,
            mycfz,
            mycs ,
            csy  ,
            csz  ,
            cfx  ,
            cfy  ,
            cfz  ,
            cs   ,
            csx)
        
        if ct.rank == ct.mainrank: print("  written: the characteristic speeds", flush=True)
    
    if ct.EnCond:
        Etot = gather(myEtot)
        if ct.rank == ct.mainrank:
            for ipnt in range(lenq):
                hdf = h5py.File(outfilename1[ipnt], "a")
                hdf.create_dataset('Etot', data=np.array(Etot[ipnt,:,:,:], dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.close()
            print("  written: the total energy Etot", flush=True)
        del(myEtot, Etot)
    
    def write(key, subArr):
        full = gather(subArr)
        if ct.rank == ct.mainrank:
            for ipnt in range(lenq):
                hdf = h5py.File(outfilename1[ipnt], "a")
                hdf.create_dataset(key, data=np.array(full[ipnt,:,:,:], dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdf.close()
    
    # For each output: zero the non-finite values, integrate in time (Equation 9), and gather and write both the rate of
    # change (eq6_) and the eigenenergy (eq9_) slabs, freeing each output as soon as it is written
    if ct.rank == ct.mainrank: print("  integrating in time and writing the " + ct.decomposition + " outputs", flush=True)
    for name in names:
        myDDT = myDDTen.pop(name)
        myDDT[~np.isfinite(myDDT)] = 0
        myen  = timeIntegral(myDDT, time, axis=-1, initial=0).astype(ct.dtype, copy=False)
        write('eq6_' + name, myDDT)
        del(myDDT)
        write('eq9_' + name, myen)
        del(myen)
        if ct.rank == ct.mainrank: print("  written: eq6_" + name + ", eq9_" + name, flush=True)
    
    if ct.rank == ct.mainrank and empty:
        for ipnt in range(lenq):
            hdf = h5py.File(outfilename1[ipnt], "a")
            for name in empty:
                for pre in ('eq6_', 'eq9_'):
                    hdf.create_dataset(pre + name, data=np.array([0], dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdf.close()
        print("  written: [0] placeholders for the invariant direction(s): " + ", ".join(empty), flush=True)
    
    ct.comm.barrier()
    
    if ct.rank == ct.mainrank:
        print("Equation 9 done: " + str(lenq) + " slab file(s) in " + dirDict["eq9"] + "\n", flush=True)
