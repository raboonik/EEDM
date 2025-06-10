from libs import prerequisites as pre
import numpy as np

def get_slice_2d(axis, nq0, xc,yc,zc,localOutpath, slq1=0, elq1=0, slq2=0, elq2=0):
    nx = len(xc)
    ny = len(yc)
    nz = len(zc)
    
    if elq1 == 0:  # Assume the single-core version
        if   axis == "xy":
            elq1 = nx
            elq2 = ny
        elif  axis == "xz":
            elq1 = nx
            elq2 = nz
        elif  axis == "yz":
            elq1 = ny
            elq2 = nz
    
    nq0   = np.array(nq0)
    _, indexes = np.unique(nq0,return_index=True)
    nq0   = nq0[np.sort(indexes)]
    lenq0 = len(nq0)
    
    if   axis == "xy":
        nq1   = nx
        nq2   = ny
        axis1 = xc
        axis2 = yc
        axis3 = zc
        if lenq0 > 0:
            if not isinstance(nq0[0],np.integer):
                nq0  = [(np.abs(nq00 - zc)).argmin() for nq00 in nq0 if (nq00 >= zc[0] and nq00 <= zc[-1])]
        lenq         = len(nq0)
        nq1i         = [slq1 for i in range(lenq)]
        nq1f         = [elq1 for i in range(lenq)]
        nq2i         = [slq2 for i in range(lenq)]
        nq2f         = [elq2 for i in range(lenq)]
        nq3i         = nq0
        nq3f         = [i+1 for i in nq0]
        outfilename1 = [localOutpath+"all_plane_"+axis+"_nz0_"+str(nq0[i])+".h5" for i in range(lenq)]
        outfilename2 = [localOutpath+"all_plane_"+axis+"_vars_nz0_"+str(nq0[0])+".h5" for i in range(lenq)]
    elif axis == "xz":
        nq1  = nx
        nq2  = nz
        axis1 = xc
        axis2 = zc
        axis3 = yc
        if lenq0 > 0:
            if not isinstance(nq0[0],np.integer):
                nq0 = [(np.abs(nq00 - yc)).argmin() for nq00 in nq0 if (nq00 >= yc[0] and nq00 <= yc[-1])]
        lenq = len(nq0)
        nq1i = [slq1 for i in range(lenq)]
        nq1f = [elq1 for i in range(lenq)]
        nq2i = nq0
        nq2f = [i+1 for i in nq0]
        nq3i = [slq2 for i in range(lenq)]
        nq3f = [elq2 for i in range(lenq)]
        outfilename1 = [localOutpath+"all_plane_"+axis+"_ny0_"+str(nq0[i])+".h5" for i in range(lenq)]
        outfilename2 = [localOutpath+"all_plane_"+axis+"_vars_ny0_"+str(nq0[0])+".h5" for i in range(lenq)]
    elif axis == "yz":
        nq1  = ny
        nq2  = nz
        axis1 = yc
        axis2 = zc
        axis3 = xc
        if lenq0 > 0:
            if not isinstance(nq0[0],np.integer):
                nq0 = [(np.abs(nq00 - xc)).argmin() for nq00 in nq0 if (nq00 >= xc[0] and nq00 <= xc[-1])]
        lenq = len(nq0)
        nq1i = nq0
        nq1f = [i+1 for i in nq0]
        nq2i = [slq1 for i in range(lenq)]
        nq2f = [elq1 for i in range(lenq)]
        nq3i = [slq2 for i in range(lenq)]
        nq3f = [elq2 for i in range(lenq)]
        outfilename1 = [localOutpath+"all_plane_"+axis+"_nx0_"+str(nq0[i])+".h5" for i in range(lenq)]
        outfilename2 = [localOutpath+"all_plane_"+axis+"_vars_nx0_"+str(nq0[0])+".h5" for i in range(lenq)]
    else:
        raise ValueError("Specify a valid plane for the variable axis!")
    
    nq0Loc = [axis3[nq00] for nq00 in nq0]
    
    return nx,ny,nz,axis1,axis2,axis3,nq0, nq0Loc, lenq, nq1, nq2, nq1i, nq1f, nq2i, nq2f, nq3i, nq3f, outfilename1, outfilename2


def get_energy_get_target_hdf(localOutpath,slicingPlane,slicingPnt,pntOrLngth):
    fid1 = [item for item in fid(localOutpath+"*.h5") if item.split("/")[-1].split("_")[2] == slicingPlane]
    targetFile = ""
    for item in fid1:
        hdf = h5py.File(item,"r")
        if pntOrLngth:
            if int(item.split("/")[-1].split("_")[-1].split(".")[0]) == slicingPnt:
                targetFile = item
                break
        else:
            if int(item.split("/")[-1].split("_")[-1].split(".")[0]) == (np.abs(slicingPnt - hdf.attrs["axis3"])).argmin():
                targetFile = item
                break
    
    if len(targetFile) == 0: raise ValueError("No HDF target file with the prescrived slicing coordinates found!")
    else: return hdf

# get_energy_plot_dict(hdf):
#     # Figure out the timestep at which the pulse starts
#     E0 = Etot[ipnt,:,:,0]
#     maxE = np.max([np.max(Etot[ipnt,:,:,i] - E0) for i in range(nt)])
#     base, exp = getBaseFloat(maxE)
#     base = round(base + 0.5)
#     ymax = base * 10**exp

#     E0 = E0[:,:,None]
#     avgE = np.average(Etot[ipnt,:,:,:] - E0)
#     for ts0 in range(nt):
#         if np.max(Etot[ipnt,0:10,0:10,ts0] - E0[0:10,0:10,:]) > avgE / 200: break
#     timestep0 = ts0
#     maxEnt = np.max([np.max(enEnt[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxDiv = np.max([np.max(enDiv[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxAlfx = np.max([np.max(enAlfx[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxAlfy = np.max([np.max(enAlfy[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxAlfz = np.max([np.max(enAlfz[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxSlox = np.max([np.max(enSlox[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxSloy = np.max([np.max(enSloy[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxSloz = np.max([np.max(enSloz[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxFasx = np.max([np.max(enFasx[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxFasy = np.max([np.max(enFasy[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxFasz = np.max([np.max(enFasz[ipnt,10:20,10:20,i]) for i in range(nt)])
#     maxE   = np.max([maxEnt, maxDiv, maxAlfx, maxSlox, maxFasx,maxAlfy, maxSloy, maxFasy,maxAlfz, maxSloz, maxFasz])
#     base, exp = getBaseFloat(maxE)
#     base  = round(base + 0.5)
#     ymax1 = base * 10**exp

#     hdf = h5py.File(outfilename1[ipnt], "a")
#     hdf.attrs['ymax']      = ymax
#     hdf.attrs['timestep0'] = timestep0

def get_slice_1d(axis, nq1, nq2, xc,yc,zc, localOutpath):
    if len(nq1) <= len(nq2):
        lenq = len(nq1)
        nq2  = [nq2[i] for i in range(lenq)]
    else:
        lenq = len(nq2)
        nq1  = [nq1[i] for i in range(lenq)]
    # Remove duplicates
    nq0 = np.array([[nq1[i],nq2[i]] for i in range(len(nq1))])
    _, indexes = np.unique(nq0, axis=0, return_index=True)     # Here underscore means ignore the first output of the double output function uniqe
    nq0 = nq0[np.sort(indexes)]
    
    lenq = len(nq0)
    
    nq1 = np.array([nq0[i][0] for i in range(lenq)])
    nq2 = np.array([nq0[i][1] for i in range(lenq)])
    
    if   axis == "x":
        if not isinstance(nq1[0],np.integer):
            nq1 = [(np.abs(nq00 - yc)).argmin() for nq00 in nq1 if (nq00 >= yc[0] and nq00 <= yc[-1])]
            nq2 = [(np.abs(nq00 - zc)).argmin() for nq00 in nq2 if (nq00 >= zc[0] and nq00 <= zc[-1])]
        if len(nq1) <= len(nq2):
            lenq = len(nq1)
            nq2  = [nq2[i] for i in range(lenq)]
        else:
            lenq = len(nq2)
            nq1  = [nq1[i] for i in range(lenq)]
        nq1i = [0 for i in range(lenq)]
        nq1f = [10000 for i in range(lenq)]
        nq2i = nq1
        nq2f = [i+1 for i in nq1]
        nq3i = nq2
        nq3f = [i+1 for i in nq2]
        nq   = len(xc)
        outfilename1 = [localOutpath+"all_axis_"+axis+"_ny0_"+str(nq1[i])+"_nz0_"+str(nq2[i])+".h5" for i in range(lenq)]
        outfilename2 = [localOutpath+"all_axis_"+axis+"_vars_ny0_"+str(nq1[i])+"_nz0_"+str(nq2[i])+".h5" for i in range(lenq)]
    elif axis == "y":
        if not isinstance(nq1[0],np.integer):
            nq1 = [(np.abs(nq00 - xc)).argmin() for nq00 in nq1 if (nq00 >= xc[0] and nq00 <= xc[-1])]
            nq2 = [(np.abs(nq00 - zc)).argmin() for nq00 in nq2 if (nq00 >= zc[0] and nq00 <= zc[-1])]
        lenq = len(nq1)
        nq1i = nq1
        nq1f = [i+1 for i in nq1]
        nq2i = [0 for i in range(lenq)]
        nq2f = [10000 for i in range(lenq)]
        nq3i = nq2
        nq3f = [i+1 for i in nq2]
        nq   = len(yc)
        outfilename1 = [localOutpath+"all_axis_"+axis+"_nx0_"+str(nq1[i])+"_nz0_"+str(nq2[i])+".h5" for i in range(lenq)]
        outfilename2 = [localOutpath+"all_axis_"+axis+"_vars_nx0_"+str(nq1[i])+"_nz0_"+str(nq2[i])+".h5" for i in range(lenq)]
    elif axis == "z":
        if not isinstance(nq1[0],np.integer):
            nq1 = [(np.abs(nq00 - xc)).argmin() for nq00 in nq1 if (nq00 >= xc[0] and nq00 <= xc[-1])]
            nq2 = [(np.abs(nq00 - yc)).argmin() for nq00 in nq2 if (nq00 >= yc[0] and nq00 <= yc[-1])]
        lenq = len(nq1)
        nq1i = nq1
        nq1f = [i+1 for i in nq1]
        nq2i = nq2
        nq2f = [i+1 for i in nq2]
        nq3i = [0 for i in range(lenq)]
        nq3f = [10000 for i in range(lenq)]
        nq   = len(zc)
        outfilename1 = [localOutpath+"all_axis_"+axis+"_nx0_"+str(nq1[i])+"_ny0_"+str(nq2[i])+".h5" for i in range(lenq)]
        outfilename2 = [localOutpath+"all_axis_"+axis+"_vars_nx0_"+str(nq1[i])+"_ny0_"+str(nq2[i])+".h5" for i in range(lenq)]
    else:
        print("Specify a valid axis!")
    
    return nq, lenq, nq1i, nq1f, nq2i, nq2f, nq3i, nq3f, outfilename1, outfilename2