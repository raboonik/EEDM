from libs import prerequisites as pre
import numpy as np

'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: Script to compute Equation 9 of Paper III (linked above)
'''

EEDMDataPath = outDirec
localOutpath = EEDMDataPath + "energySliced/"

# Make sure all the subdirs exist in localOutpath and if not create them
if rank == mainrank:
    localOutpathList    = localOutpath[1:].split("/")
    localOutpathList[0] = "/"+localOutpathList[0]
    for i in range(0,len(localOutpathList),1):
        tempDir = "/".join(localOutpathList[0:i+1]) + "/"
        err     = pre.bash("cd " + tempDir)
        if err != 0:
            pre.bash("mkdir " + tempDir)

# localOutpath is now safe to use now!
comm.barrier()


# Read the eigenenergy time derivatives, energy, and characteristic speed data
fidEn     = pre.fid(EEDMDataPath+'Etot_*.h5')
fidSp     = pre.fid(EEDMDataPath+'Speed_*.h5')
fid1      = pre.fid(EEDMDataPath+'EigenenergyDDT_*.h5')
filenames = [fid1[i].replace(EEDMDataPath, '') for i in range(len(fid1))]

nt = len(fidEn)

if rank == mainrank and (nt == 0 or len(fidSp) == 0 or len(fid1) == 0): raise ValueError("No EEDM snapshots found in "+EEDMDataPath)

comm.barrier()

dat      = pre.h5.File(fidEn[0],'r') 
divBCond = dat.attrs['divBCond']
g        = dat.attrs["g"]
xb       = np.array(dat["xc"])
yb       = np.array(dat["yc"])
zb       = np.array(dat["zc"])
dat.close()

# Although here we're only considering the full physical domain, we add the my prefix to avoid having to repeat essentially the same lines of code
# with tiny tweaks to differentiate 1D and 4D. We will modify these quantities if dim = "4D-parallelize"
nx,ny,nz,axis1,axis2,axis3,slicingPnts,slicingPntsL, mylenq, mynq1, mynq2, mynq1i, mynq1f, mynq2i, mynq2f, mynq3i, mynq3f, outfilename1, outfilename2 = \
        pre.get_slice_2d(pre.slicingPlane, pre.slicingPnts, xb,yb,zb, localOutpath)

# Note that here mylenq,mynq1,mynq2 = lenq,nq1,nq2
lenq,nq1,nq2 = mylenq,mynq1,mynq2
axes_limits  = [lenq,nq1,nq2,nt]

if rank == mainrank: print("outfilename1 = ", outfilename1)

if rank == mainrank:
    print("")
    print("The 3D data will be sliced into a " + pre.slicingPlane + "-plane at slicingPnts = n" + "xyz".strip(pre.slicingPlane) + " = " + str(slicingPnts))
    print("")

# Parallelize the first 3 axes, since we want the entire time axis to avoid having to call reshape_array_ND which is costly
parallel_axes = [0,1,2]
domDecompND   = pre.APP.domainDecomposeND(size, axes_limits, parallel_axes)
slPnts        = domDecompND.slq[0][rank]
elPnts        = domDecompND.elq[0][rank]
slq1          = domDecompND.slq[1][rank]
elq1          = domDecompND.elq[1][rank]
slq2          = domDecompND.slq[2][rank]
elq2          = domDecompND.elq[2][rank]

if rank == 0:
    print("size = "                 , size)
    print("axes_limits = "          , axes_limits)
    print("parallel_axes = "        , parallel_axes)
    print("domainDecomposeND = "    , domDecompND)
    print("domainDecomposeND.slq = ", domDecompND.slq)
    print("domainDecomposeND.elq = ", domDecompND.elq)

# Get the sub-sub-spatial slicing indeces for each proc
if   pre.slicingPlane == "xy":
    _,_,_,_,_,_,mynq0,_, mylenq, mynq1, mynq2, mynq1i,   \
    mynq1f, mynq2i, mynq2f, mynq3i, mynq3f, _,_ = \
    pre.get_slice_2d(pre.slicingPlane, slicingPnts[slPnts:elPnts], xb[slq1:elq1],yb[slq2:elq2],zb, localOutpath, slq1,elq1,slq2,elq2)
elif pre.slicingPlane == "xz":
    _,_,_,_,_,_,mynq0,_, mylenq, mynq1, mynq2, mynq1i,   \
    mynq1f, mynq2i, mynq2f, mynq3i, mynq3f, _,_ = \
    pre.get_slice_2d(pre.slicingPlane, slicingPnts[slPnts:elPnts], xb[slq1:elq1],yb,zb[slq2:elq2], localOutpath, slq1,elq1,slq2,elq2)
elif pre.slicingPlane == "yz":
    _,_,_,_,_,_,mynq0,_, mylenq, mynq1, mynq2, mynq1i,   \
    mynq1f, mynq2i, mynq2f, mynq3i, mynq3f, _,_ = \
    pre.get_slice_2d(pre.slicingPlane, slicingPnts[slPnts:elPnts], xb,yb[slq1:elq1],zb[slq2:elq2], localOutpath, slq1,elq1,slq2,elq2)

myshape = domDecompND.mynq[:,rank]

print("rank, myshape, mynq1i, mynq1f, mynq2i, mynq2f, mynq3i, mynq3f = ", rank, myshape, mynq1i, mynq1f, mynq2i, mynq2f, mynq3i, mynq3f)

# The benefit of using the my prefex from the get-go is that we don't have to repeat the same lines of code for 1D and 4D separately
myDDTenDivx  = np.zeros(myshape)
myDDTenDivy  = np.zeros(myshape)
myDDTenDivz  = np.zeros(myshape)
myDDTenEntx  = np.zeros(myshape)
myDDTenEnty  = np.zeros(myshape)
myDDTenEntz  = np.zeros(myshape)
myDDTenGra   = np.zeros(myshape)
myDDTenAlfx  = np.zeros(myshape)
myDDTenSlox  = np.zeros(myshape)
myDDTenFasx  = np.zeros(myshape)
myDDTenAlfy  = np.zeros(myshape)
myDDTenSloy  = np.zeros(myshape)
myDDTenFasy  = np.zeros(myshape)
myDDTenAlfz  = np.zeros(myshape)
myDDTenSloz  = np.zeros(myshape)
myDDTenFasz  = np.zeros(myshape)
myax         = np.zeros(myshape)
myay         = np.zeros(myshape)
myaz         = np.zeros(myshape)
mycsx        = np.zeros(myshape)
mycsy        = np.zeros(myshape)
mycsz        = np.zeros(myshape)
mycfx        = np.zeros(myshape)
mycfy        = np.zeros(myshape)
mycfz        = np.zeros(myshape)
mycs         = np.zeros(myshape)
myEtot       = np.zeros(myshape)

if divBCond: myDDTenDiv1 = np.zeros(myshape)

if pre.mode == "XYZUpDownSeparated":
    myDDTenAlfx_2 = np.zeros(myshape)
    myDDTenSlox_2 = np.zeros(myshape)
    myDDTenFasx_2 = np.zeros(myshape)
    myDDTenAlfy_2 = np.zeros(myshape)
    myDDTenSloy_2 = np.zeros(myshape)
    myDDTenFasy_2 = np.zeros(myshape)
    myDDTenAlfz_2 = np.zeros(myshape)
    myDDTenSloz_2 = np.zeros(myshape)
    myDDTenFasz_2 = np.zeros(myshape)



# Since t is not parallelized, we can simply read time from HDF times for each rank individually
time        = np.zeros(nt)
maxDivB     = np.zeros(nt)
meanDivB    = np.zeros(nt)
stdDivB     = np.zeros(nt)
maxEntropy  = np.zeros(nt)
meanEntropy = np.zeros(nt)
stdEntropy  = np.zeros(nt)

for i in range(nt):
    filename = filenames[i]
    if rank == mainrank: print("Reading", filename)
    dat   = pre.h5.File(open(fid1[i],  'rb'), 'r')
    datSp = pre.h5.File(open(fidSp[i], 'rb'), 'r')
    datEn = pre.h5.File(open(fidEn[i], 'rb'), 'r')
    
    time[i]        = float(  dat.attrs['time'          ])
    maxDivB[i]     = float(datEn.attrs['maxAbsDivB'    ])
    meanDivB[i]    = float(datEn.attrs['meanAbsDivB'   ])
    stdDivB[i]     = float(datEn.attrs['stdAbsDivB'    ])
    maxEntropy[i]  = float(datEn.attrs['maxAbsEntropy' ])
    meanEntropy[i] = float(datEn.attrs['meanAbsEntropy'])
    stdEntropy[i]  = float(datEn.attrs['stdAbsEntropy' ])
    
    for ipnt in range(mylenq):
        myDDTenDivx[ ipnt,:,:,i] = np.squeeze(dat["m1_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenDivy[ ipnt,:,:,i] = np.squeeze(dat["m1_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenDivz[ ipnt,:,:,i] = np.squeeze(dat["m1_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenEntx[ ipnt,:,:,i] = np.squeeze(dat["m2_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenEnty[ ipnt,:,:,i] = np.squeeze(dat["m2_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenEntz[ ipnt,:,:,i] = np.squeeze(dat["m2_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenAlfx[ipnt,:,:,i]  = np.squeeze(dat["m3_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenSlox[ipnt,:,:,i]  = np.squeeze(dat["m5_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenFasx[ipnt,:,:,i]  = np.squeeze(dat["m7_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenAlfy[ipnt,:,:,i]  = np.squeeze(dat["m3_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenSloy[ipnt,:,:,i]  = np.squeeze(dat["m5_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenFasy[ipnt,:,:,i]  = np.squeeze(dat["m7_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenAlfz[ipnt,:,:,i]  = np.squeeze(dat["m3_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenSloz[ipnt,:,:,i]  = np.squeeze(dat["m5_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myDDTenFasz[ipnt,:,:,i]  = np.squeeze(dat["m7_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        
        if pre.mode == "XYZUpDownSeparated":
            myDDTenAlfx_2[ipnt,:,:,i] = np.squeeze(dat["m4_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenSlox_2[ipnt,:,:,i] = np.squeeze(dat["m6_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenFasx_2[ipnt,:,:,i] = np.squeeze(dat["m8_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenAlfy_2[ipnt,:,:,i] = np.squeeze(dat["m4_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenSloy_2[ipnt,:,:,i] = np.squeeze(dat["m6_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenFasy_2[ipnt,:,:,i] = np.squeeze(dat["m8_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenAlfz_2[ipnt,:,:,i] = np.squeeze(dat["m4_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenSloz_2[ipnt,:,:,i] = np.squeeze(dat["m6_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenFasz_2[ipnt,:,:,i] = np.squeeze(dat["m8_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        elif pre.mode == "revFor_combined":
            myDDTenAlfx[ipnt,:,:,i] = myDDTenAlfx[ipnt,:,:,i] + np.squeeze(dat["m4_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenSlox[ipnt,:,:,i] = myDDTenSlox[ipnt,:,:,i] + np.squeeze(dat["m6_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenFasx[ipnt,:,:,i] = myDDTenFasx[ipnt,:,:,i] + np.squeeze(dat["m8_x"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenAlfy[ipnt,:,:,i] = myDDTenAlfy[ipnt,:,:,i] + np.squeeze(dat["m4_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenSloy[ipnt,:,:,i] = myDDTenSloy[ipnt,:,:,i] + np.squeeze(dat["m6_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenFasy[ipnt,:,:,i] = myDDTenFasy[ipnt,:,:,i] + np.squeeze(dat["m8_y"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenAlfz[ipnt,:,:,i] = myDDTenAlfz[ipnt,:,:,i] + np.squeeze(dat["m4_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenSloz[ipnt,:,:,i] = myDDTenSloz[ipnt,:,:,i] + np.squeeze(dat["m6_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
            myDDTenFasz[ipnt,:,:,i] = myDDTenFasz[ipnt,:,:,i] + np.squeeze(dat["m8_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])    
        
        myax[  ipnt,:,:,i]      = np.squeeze(   datSp["ax"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myay[  ipnt,:,:,i]      = np.squeeze(   datSp["ay"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myaz[  ipnt,:,:,i]      = np.squeeze(   datSp["az"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        mycsx[ ipnt,:,:,i]      = np.squeeze(  datSp["csx"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        mycsy[ ipnt,:,:,i]      = np.squeeze(  datSp["csy"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        mycsz[ ipnt,:,:,i]      = np.squeeze(  datSp["csz"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        mycfx[ ipnt,:,:,i]      = np.squeeze(  datSp["cfx"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]]) 
        mycfy[ ipnt,:,:,i]      = np.squeeze(  datSp["cfy"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        mycfz[ ipnt,:,:,i]      = np.squeeze(  datSp["cfz"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        myEtot[ipnt,:,:,i]      = np.squeeze(  datEn["Kin"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]] +
                                               datEn["Mag"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]] +
                                               datEn["Int"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        mycs[  ipnt,:,:,i]      = np.sqrt(mycfz[ipnt,:,:,i]**2 + mycsz[ipnt,:,:,i]**2 - (myax[ipnt,:,:,i]**2 + myay[ipnt,:,:,i]**2 + myaz[ipnt,:,:,i]**2))
        if g > 0: 
            myEtot[ipnt,:,:,i]  = np.squeeze(datEn["Grv"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]]) + myEtot[ipnt,:,:,i]
            myDDTenGra[ ipnt,:,:,i] = np.squeeze(dat["m9_z"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
        if divBCond: 
            myDDTenDiv1[ ipnt,:,:,i] = np.squeeze(dat["m1_err"][mynq1i[ipnt]:mynq1f[ipnt],mynq2i[ipnt]:mynq2f[ipnt],mynq3i[ipnt]:mynq3f[ipnt]])
    
    dat.close()
    datSp.close()
    datEn.close()

# Gather everything on mainrank to export
if rank == mainrank:
    print("Gathering everything on mainrank and writing to ouput h5 files...\n")
    print("Gathering the velocity components...\n")

ax  = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myax )
ay  = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myay )
az  = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myaz )

if rank == mainrank:
    print("aq gatehred, starting to write to file!\n")
    timesteps = (np.roll(time,-1,axis=0) - time)[0:-1]
    for ipnt in range(lenq):
        print("Writing 3D-2D sliced data to "+outfilename1[ipnt]+"\n")
        hdf = pre.h5.File(outfilename1[ipnt], "w")
        hdf.attrs['slicingPnt']   = slicingPnts[ipnt]
        hdf.attrs['slicingPntL']  = slicingPntsL[ipnt]
        hdf.attrs['slicingPlane'] = pre.slicingPlane
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
        hdf.attrs['divBCond']     = divBCond
        hdf.create_dataset('xc'          ,   data=np.array(xb,              dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('yc'          ,   data=np.array(yb,              dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('zc'          ,   data=np.array(zb,              dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('time'        ,   data=np.array(time,            dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('maxDivB'     ,   data=np.array(maxDivB,         dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('meanDivB'    ,   data=np.array(meanDivB,        dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('stdDivB'     ,   data=np.array(stdDivB,         dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('maxEntropy'  ,   data=np.array(maxEntropy,      dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('meanEntropy' ,   data=np.array(meanEntropy,     dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('stdEntropy'  ,   data=np.array(stdEntropy,      dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('timesteps'   ,   data=np.array(timesteps,       dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('ax'          ,   data=np.array(ax[ipnt,:,:,:] , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('ay'          ,   data=np.array(ay[ipnt,:,:,:] , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('az'          ,   data=np.array(az[ipnt,:,:,:] , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(myax,
    myay,
    myaz,
    ax  ,
    ay  ,
    az)

csx = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, mycsx)
csy = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, mycsy)
csz = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, mycsz)
cfx = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, mycfx)
cfy = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, mycfy)
cfz = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, mycfz)
cs  = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, mycs )

if rank == mainrank:
    print("csq,cfq,cs gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset('csx',         data=np.array(csx[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('csy',         data=np.array(csy[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('csz',         data=np.array(csz[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('cfx',         data=np.array(cfx[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('cfy',         data=np.array(cfy[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('cfz',         data=np.array(cfz[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('cs' ,         data=np.array(cs[ipnt,:,:,:] , dtype='float64'), compression='gzip', compression_opts=9)
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

if rank == mainrank: print("Writing velocity components to file done!\n")


myenDivx  = np.zeros(myshape)
myenDivy  = np.zeros(myshape)
myenDivz  = np.zeros(myshape)
if divBCond: myenDiv1  = np.zeros(myshape)
myenEntx  = np.zeros(myshape)
myenEnty  = np.zeros(myshape)
myenEntz  = np.zeros(myshape)
if g > 0: myenGra  = np.zeros(myshape)
myenAlfx = np.zeros(myshape)
myenAlfy = np.zeros(myshape)
myenAlfz = np.zeros(myshape)
myenSlox = np.zeros(myshape)
myenSloy = np.zeros(myshape)
myenSloz = np.zeros(myshape)
myenFasx = np.zeros(myshape)
myenFasy = np.zeros(myshape)
myenFasz = np.zeros(myshape)

if pre.mode == "XYZUpDownSeparated":
    myenAlfx_2 = np.zeros(myshape)
    myenAlfy_2 = np.zeros(myshape)
    myenAlfz_2 = np.zeros(myshape)
    myenSlox_2 = np.zeros(myshape)
    myenSloy_2 = np.zeros(myshape)
    myenSloz_2 = np.zeros(myshape)
    myenFasx_2 = np.zeros(myshape)
    myenFasy_2 = np.zeros(myshape)
    myenFasz_2 = np.zeros(myshape)
    

if rank == mainrank: print("\nComputing the time integrals...\n")
for ipnt in range(myshape[0]):
    for k1 in range(myshape[1]):
        for k2 in range(myshape[2]):
            # DivB
            dat                      = myDDTenDivx[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenDivx[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            dat                      = myDDTenDivy[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenDivy[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            dat                      = myDDTenDivz[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenDivz[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            # DivB1
            if divBCond:
                dat                      = myDDTenDiv1[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                myenDiv1[ipnt,k1,k2,:]    = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            # Ent
            dat                      = myDDTenEntx[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenEntx[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            dat                      = myDDTenEnty[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenEnty[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            dat                      = myDDTenEntz[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenEntz[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            # Grav
            if g > 0:
                dat                      = myDDTenGra[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                myenGra[ipnt,k1,k2,:]    = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            # Alfven
            dat                      = myDDTenAlfx[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenAlfx[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            
            dat                      = myDDTenAlfy[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenAlfy[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            
            dat                      = myDDTenAlfz[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenAlfz[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            
            # Slow
            dat                      = myDDTenSlox[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            mask                     = np.isinf(dat)
            dat[mask]                = 0
            myenSlox[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            
            dat                      = myDDTenSloy[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            mask                     = np.isinf(dat)
            dat[mask]                = 0
            myenSloy[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            
            dat                      = myDDTenSloz[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            mask                     = np.isinf(dat)
            dat[mask]                = 0
            myenSloz[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            
            # Fast
            dat                      = myDDTenFasx[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenFasx[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            
            dat                      = myDDTenFasy[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenFasy[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            
            dat                      = myDDTenFasz[ipnt,k1,k2,:]
            mask                     = np.isnan(dat)
            dat[mask]                = 0
            myenFasz[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
            
            if pre.mode == "XYZUpDownSeparated":
                # Alfven
                dat                      = myDDTenAlfx_2[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                myenAlfx_2[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
                            
                dat                      = myDDTenAlfy_2[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                myenAlfy_2[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
                            
                dat                      = myDDTenAlfz_2[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                myenAlfz_2[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
                            
                # Slow
                dat                      = myDDTenSlox_2[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                mask                     = np.isinf(dat)
                dat[mask]                = 0
                myenSlox_2[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
                            
                dat                      = myDDTenSloy_2[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                mask                     = np.isinf(dat)
                dat[mask]                = 0
                myenSloy_2[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
                            
                dat                      = myDDTenSloz_2[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                mask                     = np.isinf(dat)
                dat[mask]                = 0
                myenSloz_2[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
                            
                # Fast
                dat                      = myDDTenFasx_2[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                myenFasx_2[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
                            
                dat                      = myDDTenFasy_2[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                myenFasy_2[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
                            
                dat                      = myDDTenFasz_2[ipnt,k1,k2,:]
                mask                     = np.isnan(dat)
                dat[mask]                = 0
                myenFasz_2[ipnt,k1,k2,:]   = pre.spy.integrate.cumtrapz(dat,time,axis=0,initial=0)
                
mask,dat = 0,0
del(mask,
    dat)


if rank == mainrank: print("Gathering the rate of energy change components...\n")
DDTenDivx       = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenDivx)
DDTenEntx       = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenEntx)
DDTenAlfx       = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenAlfx)
if pre.mode == "XYZUpDownSeparated":
    DDTenAlfx_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenAlfx_2)

if divBCond: 
    DDTenDiv1   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenDiv1)
if g > 0:
    DDTenGra    = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenGra)

if rank == mainrank:
    print("X-components of the modes 1, 2, and 3 of Equations6 gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset('eq6_m1_x',data=np.array(DDTenDivx[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq6_m2_x',data=np.array(DDTenEntx[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9) 
        hdf.create_dataset('eq6_m3_x',data=np.array(DDTenAlfx[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        if pre.mode == "XYZUpDownSeparated":
            hdf.create_dataset('eq6_m4_x'  ,data=np.array(DDTenAlfx_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        if divBCond: 
            hdf.create_dataset('eq6_m1_err',data=np.array(DDTenDiv1[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        if g > 0: 
            hdf.create_dataset('eq6_m9_z'  ,data=np.array(DDTenGra[ipnt,:,:,:]   , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(myDDTenEntx,
    myDDTenAlfx,
    DDTenAlfx  ,
    myDDTenDivx)

if pre.mode == "XYZUpDownSeparated": 
    del(myDDTenAlfx_2,
        DDTenAlfx_2  ,
        DDTenEntx    ,
        DDTenDivx)

if divBCond:                         
    del(myDDTenDiv1,
        DDTenDiv1)

if g > 0:                            
    del(myDDTenGra,
        DDTenGra)

DDTenDivy   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenDivy)
DDTenDivz   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenDivz)
DDTenEnty   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenEnty)
DDTenEntz   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenEntz)

if rank == mainrank:
    print("eq6_m1_y,eq6_m1_z,eq6_m2_y,eq6_m2_z gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset('eq6_m1_y', data=np.array(DDTenDivy[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq6_m1_z', data=np.array(DDTenDivz[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq6_m2_y', data=np.array(DDTenEnty[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq6_m2_z', data=np.array(DDTenEntz[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(DDTenDivy  ,
    DDTenDivz  ,
    DDTenEnty  ,
    DDTenEntz  ,
    myDDTenDivy,
    myDDTenDivz,
    myDDTenEnty,
    myDDTenEntz)

DDTenSlox   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenSlox)
DDTenFasx   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenFasx)
DDTenAlfy   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenAlfy)

if pre.mode == "XYZUpDownSeparated": 
    DDTenSlox_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenSlox_2)
    DDTenFasx_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenFasx_2)
    DDTenAlfy_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenAlfy_2)

if rank == mainrank:
    print("eq6_m5_x,eq6_m7_x,eq6_m3_y gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset('eq6_m5_x'    , data=np.array(DDTenSlox[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq6_m7_x'    , data=np.array(DDTenFasx[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq6_m3_y'    , data=np.array(DDTenAlfy[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        if pre.mode == "XYZUpDownSeparated":
            hdf.create_dataset('eq6_m6_x', data=np.array(DDTenSlox_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq6_m8_x', data=np.array(DDTenFasx_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq6_m4_y', data=np.array(DDTenAlfy_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(DDTenSlox  ,
    DDTenFasx  ,
    DDTenAlfy  ,
    myDDTenSlox,
    myDDTenFasx,
    myDDTenAlfy)

if pre.mode == "XYZUpDownSeparated":
    del(DDTenSlox_2  ,
        DDTenFasx_2  ,
        DDTenAlfy_2  ,
        myDDTenSlox_2,
        myDDTenFasx_2,
        myDDTenAlfy_2)

DDTenSloy   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenSloy)
DDTenFasy   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenFasy)
DDTenAlfz   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenAlfz)

if pre.mode == "XYZUpDownSeparated":
    DDTenSloy_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenSloy_2)
    DDTenFasy_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenFasy_2)
    DDTenAlfz_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenAlfz_2)

if rank == mainrank:
    print("eq6_m5_y,eq6_m7_y,eq6_m3_z gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset('eq6_m5_y'    , data=np.array(DDTenSloy[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq6_m7_y'    , data=np.array(DDTenFasy[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq6_m3_z'    , data=np.array(DDTenAlfz[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        if pre.mode == "XYZUpDownSeparated":
            hdf.create_dataset('eq6_m6_y', data=np.array(DDTenSloy_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq6_m8_y', data=np.array(DDTenFasy_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq6_m4_z', data=np.array(DDTenAlfz_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(DDTenSloy  ,
    DDTenFasy  ,
    DDTenAlfz  ,
    myDDTenSloy,
    myDDTenFasy,
    myDDTenAlfz)

if pre.mode == "XYZUpDownSeparated":
    del(DDTenSloy_2  ,
        DDTenFasy_2  ,
        DDTenAlfz_2  ,
        myDDTenSloy_2,
        myDDTenFasy_2,
        myDDTenAlfz_2)

DDTenSloz   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenSloz)
DDTenFasz   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenFasz)

if pre.mode == "XYZUpDownSeparated":
    DDTenSloz_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenSloz_2)
    DDTenFasz_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myDDTenFasz_2)

if rank == mainrank:
    print("eq6_m5_z,eq6_m7_z gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset('eq6_m5_z'    , data=np.array(DDTenSloz[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq6_m7_z'    , data=np.array(DDTenFasz[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        if pre.mode == "XYZUpDownSeparated":
            hdf.create_dataset('eq6_m6_z', data=np.array(DDTenSloz_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq6_m8_z', data=np.array(DDTenFasz_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(DDTenSloz  ,
    DDTenFasz  ,
    myDDTenSloz,
    myDDTenFasz)

if pre.mode == "XYZUpDownSeparated":
    del(DDTenSloz_2  ,
        DDTenFasz_2  ,
        myDDTenSloz_2,
        myDDTenFasz_2)
    
if rank == mainrank: print("Writing the rate of energy change components to file done!\n")


if rank == mainrank: print("Gathering the energy components...\n")
Etot        = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myEtot     )
enDivx      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenDivx   ) 
enEntx      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenEntx   ) 

if divBCond: 
    enDiv1  = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenDiv1   ) 
if g > 0: 
    enGra   = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenGra    )
 
if rank == mainrank:
    print("Etot,eq9_m1_x,eq9_m2_x gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset(    'Etot'      , data=np.array(Etot[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset(    'eq9_m1_x'  , data=np.array(enDivx[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset(    'eq9_m2_x'  , data=np.array(enEntx[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        if divBCond:
            hdf.create_dataset('eq9_m1_err', data=np.array(enDiv1[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        if g > 0:
            hdf.create_dataset('eq9_m9_z'  , data=np.array(enGra[ipnt,:,:,:] , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(myEtot  ,
    myenEntx,
    myenDivx)

if divBCond: 
    del(myenDiv1,
        enDiv1  ,
        enDivx  ,
        enEntx  , 
        Etot)

if g > 0: 
    del(myenGra,
        enGra)

enDivy      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenDivy   ) 
enDivz      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenDivz   ) 
enEnty      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenEnty   ) 
enEntz      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenEntz   ) 

if rank == mainrank:
    print("eq9_m1_y,eq9_m1_z,eq9_m2_y,eq9_m2_z gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset('eq9_m1_y', data=np.array(enDivy[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq9_m1_z', data=np.array(enDivz[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq9_m2_y', data=np.array(enEnty[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq9_m2_z', data=np.array(enEntz[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(enDivy  ,
    enDivz  ,
    enEnty  ,
    enEntz  ,
    myenDivy,
    myenDivz,
    myenEnty,
    myenEntz)

enAlfx      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenAlfx   ) 
enAlfy      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenAlfy   ) 
enAlfz      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenAlfz   ) 

if pre.mode == "XYZUpDownSeparated":
    enAlfx_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenAlfx_2) 
    enAlfy_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenAlfy_2) 
    enAlfz_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenAlfz_2) 

if rank == mainrank:
    print("The Alfven terms in Equation 9 gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset('eq9_m3_x'    , data=np.array(enAlfx[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq9_m3_y'    , data=np.array(enAlfy[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq9_m3_z'    , data=np.array(enAlfz[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        if pre.mode == "XYZUpDownSeparated":
            hdf.create_dataset('eq9_m4_x', data=np.array(enAlfx_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq9_m4_y', data=np.array(enAlfy_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq9_m4_z', data=np.array(enAlfz_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(enAlfx  ,
    myenAlfx,
    enAlfy  ,
    myenAlfy,
    enAlfz  ,
    myenAlfz)

if pre.mode == "XYZUpDownSeparated":
    del(enAlfx_2  ,
        myenAlfx_2,
        enAlfy_2  ,
        myenAlfy_2,
        enAlfz_2  ,
        myenAlfz_2)

enSlox      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenSlox   ) 
enSloy      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenSloy   ) 
enSloz      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenSloz   ) 

if pre.mode == "XYZUpDownSeparated":
    enSlox_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenSlox_2) 
    enSloy_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenSloy_2) 
    enSloz_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenSloz_2) 

if rank == mainrank:
    print("The slow terms in Equation 9 gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset('eq9_m5_x'    , data=np.array(enSlox[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq9_m5_y'    , data=np.array(enSloy[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq9_m5_z'    , data=np.array(enSloz[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        if pre.mode == "XYZUpDownSeparated":
            hdf.create_dataset('eq9_m6_x', data=np.array(enSlox_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq9_m6_y', data=np.array(enSloy_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq9_m6_z', data=np.array(enSloz_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(myenSlox,
    myenSloy,
    myenSloz,
    enSlox  ,
    enSloy  ,
    enSloz)

if pre.mode == "XYZUpDownSeparated":
    del(myenSlox_2,
        myenSloy_2,
        myenSloz_2,
        enSlox_2  ,
        enSloy_2  ,
        enSloz_2)


enFasx      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenFasx   ) 
enFasy      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenFasy   ) 
enFasz      = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenFasz   ) 

if pre.mode == "XYZUpDownSeparated":
    enFasx_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenFasx_2) 
    enFasy_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenFasy_2) 
    enFasz_2 = pre.APP.gather_array_ND(comm, rank, mainrank, domDecompND, myenFasz_2) 

if rank == mainrank:
    print("The fast terms in Equation 9 gatehred, starting to write to file!\n")
    for ipnt in range(lenq):
        hdf = pre.h5.File(outfilename1[ipnt], "a")
        hdf.create_dataset('eq9_m7_x'    , data=np.array(enFasx[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq9_m7_y'    , data=np.array(enFasy[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        hdf.create_dataset('eq9_m7_z'    , data=np.array(enFasz[ipnt,:,:,:]  , dtype='float64'), compression='gzip', compression_opts=9)
        if pre.mode == "XYZUpDownSeparated":
            hdf.create_dataset('eq9_m8_x', data=np.array(enFasx_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq9_m8_y', data=np.array(enFasy_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
            hdf.create_dataset('eq9_m8_z', data=np.array(enFasz_2[ipnt,:,:,:], dtype='float64'), compression='gzip', compression_opts=9)
        hdf.close()

del(enFasx  ,
    myenFasx,
    enFasy  ,
    myenFasy,
    enFasz  ,
    myenFasz)

if pre.mode == "XYZUpDownSeparated":
    del(enFasx_2  ,
        myenFasx_2,
        enFasy_2  ,
        myenFasy_2,
        enFasz_2  ,
        myenFasz_2)


comm.barrier()

if rank == mainrank:
    print("\nEquation 9 computed and saved in " + localOutpath + ".")