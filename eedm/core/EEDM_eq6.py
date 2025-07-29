'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: Script to compute Equations 6 of Paper III (linked above)
'''

import numpy as np
import sys
import h5py

from .. import io
from .. import utils
from .. import context as ct
from .. import methods
from .. import const
from .. import decorators

import settings as set


@decorators.memoize
def run(dirDict):
    # Import gamma, g, and mu0 for ease of use
    gamma = set.gamma
    g     = set.g
    mu0   = const.mu0
    
    # IniInitialize the reading of the simulation data
    readObj = io.reader.EEDM_reader()
    
    # Initial parallelization scheme based on the number of snapshots
    slt = int(ct.rank       * readObj.nt / ct.size)
    elt = int((ct.rank + 1) * readObj.nt / ct.size)
    
    if ct.rank == 0:
        split_sizes = np.zeros(ct.size,'int')
    else:
        split_sizes = None
    
    ct.comm.Gather(sendbuf=np.array(elt - slt,dtype='int'), recvbuf=split_sizes, root=0)
    split_sizes = ct.comm.bcast(split_sizes,root=0)
    
    # Get the filenames
    filenames = [readObj.feed[i].replace(set.datapath, '') for i in range(slt,elt)]
    
    # Compute the eigenenergy time derivatives (Equations 6 of Paper III)
    for i in range(slt,elt):
        filename                       = filenames[i - slt]
        rho, vx, vy, vz, bx, by, bz, p = readObj.data(i)
        time                           = readObj.time
        
        print("rank = ", ct.rank, " Reading", filename + ", simulation time = ", time)
        
        # Compute vsq and save for later
        vsq = vx**2 + vy**2 + vz**2
        
        if ct.EnCond:
            # Save the kinetic, magnetic, internal, and gravitational energies
            hdfEtot =  h5py.File(dirDict["energy"] + "Etot_" + filename[0:-4] + ".h5", 'w')
            
            hdfEtot.create_dataset('Kin', data=np.array(0.5 * rho * vsq                    , dtype='float64'), compression='gzip', compression_opts=9) 
            hdfEtot.create_dataset('Mag', data=np.array(0.5 * (bx**2 + by**2 + bz**2) / mu0, dtype='float64'), compression='gzip', compression_opts=9) 
            hdfEtot.create_dataset('Int', data=np.array(p / (gamma - 1)                    , dtype='float64'), compression='gzip', compression_opts=9) 
            if g == 0:
                hdfEtot.create_dataset('Grv', data=np.array([0]                 ,dtype='float64'), compression='gzip', compression_opts=9) 
            else:
                hdfEtot.create_dataset('Grv', data=np.array(rho * g * readObj.zc,dtype='float64'), compression='gzip', compression_opts=9)
            
            hdfEtot.close()
        
        
        # Save the grid and parameters in a single file
        
        if i == 0:
            hdfParam =  h5py.File(dirDict["parent"] + "grid_params.h5", 'w')
            hdfParam.create_dataset('xc',     data=np.array(readObj.xc    ,dtype='float64'), compression='gzip', compression_opts=9)
            hdfParam.create_dataset('yc',     data=np.array(readObj.yc    ,dtype='float64'), compression='gzip', compression_opts=9)
            hdfParam.create_dataset('zc',     data=np.array(readObj.zc    ,dtype='float64'), compression='gzip', compression_opts=9)
            hdfParam.create_dataset('framec', data=np.array(readObj.framec,dtype='float64'), compression='gzip', compression_opts=9)
            if not np.any(readObj.xb == None):
                hdfParam.create_dataset('frameb', data=np.array(readObj.frameb,dtype='float64'), compression='gzip', compression_opts=9)
                hdfParam.create_dataset('xb',     data=np.array(readObj.xb    ,dtype='float64'), compression='gzip', compression_opts=9)
                hdfParam.create_dataset('yb',     data=np.array(readObj.yb    ,dtype='float64'), compression='gzip', compression_opts=9)
                hdfParam.create_dataset('zb',     data=np.array(readObj.zb    ,dtype='float64'), compression='gzip', compression_opts=9)
            
            hdfParam.attrs['mode']     = set.mode
            hdfParam.attrs['gamma'   ] = gamma
            hdfParam.attrs['mu0'     ] = mu0
            hdfParam.attrs['g']        = g
            hdfParam.attrs['gCond']    = g > 0
            hdfParam.close()
        
        # Let's roughly keep track of how much memory we're using
        mem = sys.getsizeof(readObj) / const.gig
        mem = mem + (sys.getsizeof(vx )+sys.getsizeof(vy)+sys.getsizeof(vz)+
                     sys.getsizeof(bx )+sys.getsizeof(by)+sys.getsizeof(bz)+
                     sys.getsizeof(rho)+sys.getsizeof(vsq)) / const.gig
        print("rank = ", ct.rank, " Memory used before computing the derivatives = ", mem, " Gb\n")
        
        # Compute the field derivatives
        BxDx = methods.diff.partial(bx,0,readObj.xc)
        BxDy = methods.diff.partial(bx,1,readObj.yc)
        BxDz = methods.diff.partial(bx,2,readObj.zc)
        ByDx = methods.diff.partial(by,0,readObj.xc)
        ByDy = methods.diff.partial(by,1,readObj.yc)
        ByDz = methods.diff.partial(by,2,readObj.zc)
        BzDx = methods.diff.partial(bz,0,readObj.xc)
        BzDy = methods.diff.partial(bz,1,readObj.yc)
        BzDz = methods.diff.partial(bz,2,readObj.zc)
        
        sqmurho = np.sqrt(rho * mu0)
        ax      = bx / sqmurho
        ay      = by / sqmurho
        az      = bz / sqmurho
        
        mem = mem + (sys.getsizeof(BxDx)+sys.getsizeof(BxDy)+sys.getsizeof(BxDz)+
                    sys.getsizeof(ByDx)+sys.getsizeof(ByDy)+sys.getsizeof(ByDz)+
                    sys.getsizeof(BzDx)+sys.getsizeof(BzDy)+sys.getsizeof(BzDz)+
                    sys.getsizeof(ax)+sys.getsizeof(ay)+sys.getsizeof(az)+
                    -(sys.getsizeof(bx)+sys.getsizeof(by)+sys.getsizeof(bz))+
                    sys.getsizeof(sqmurho)) / const.gig
        print("rank = ", ct.rank, " Memory used before computing the characteristic speeds = ", mem, " Gb\n")
        
        # No longer need B
        del(bx,
            by,
            bz)
        
        # If necessary, compute and save divB and the polytropic k, where dk/dt|_Lagrangian = 0
        if ct.DbCond or ct.PkCond: 
            hdfExtra =  h5py.File(dirDict["extra"] + "Extras_"+filename[0:-4] + ".h5", 'w')
        
        if ct.PkCond:
            poly_k                           = p / rho**gamma            
            hdfExtra.create_dataset('poly_k', data=np.array(poly_k,dtype='float64'), compression='gzip', compression_opts=9)
            hdfExtra.attrs['maxAbsEntropy' ] = np.max( np.abs(poly_k))
            hdfExtra.attrs['meanAbsEntropy'] = np.mean(np.abs(poly_k))
            hdfExtra.attrs['stdAbsEntropy' ] = np.std( np.abs(poly_k))
            del(poly_k)
        
        if ct.DbCond:
            divB                          = BxDx + ByDy + BzDz
            hdfExtra.create_dataset('divB', data=np.array(divB,dtype='float64'), compression='gzip', compression_opts=9)
            hdfExtra.attrs['maxAbsDivB' ] = np.max(np.abs(divB))
            hdfExtra.attrs['meanAbsDivB'] = np.mean(np.abs(divB))
            hdfExtra.attrs['stdAbsDivB' ] = np.std(np.abs(divB))
            del(divB)
        
        if ct.DbCond or ct.PkCond: 
            hdfExtra.close()
        
        
        # Compute the characteristic speeds
        
        # Sound speed
        c   = np.sqrt(gamma * p / rho)
        
        a2c2 = ax**2 + ay**2 + az**2 + c**2 # = csq**2 + cfq**2
        
        # x-component of the slow and fast speeds
        csx  = a2c2**2 - (2 * ax * c)**2
        # Let's make sure this is positive
        utils.filter.SQRTnegFilter(csx)
        
        cfx = np.sqrt(a2c2 + np.sqrt(csx)) / np.sqrt(2)
        
        csx = a2c2 - np.sqrt(csx)
        utils.filter.SQRTnegFilter(csx)
        
        csx = np.sqrt(csx) / np.sqrt(2)
        
        # y-component of the slow and fast speeds
        csy  = a2c2**2 - (2 * ay * c)**2
        utils.filter.SQRTnegFilter(csy)
        
        cfy = np.sqrt(a2c2 + np.sqrt(csy)) / np.sqrt(2)
        
        csy = a2c2 - np.sqrt(csy)
        utils.filter.SQRTnegFilter(csy)
        
        csy = np.sqrt(csy) / np.sqrt(2)
        
        # z-component of the slow and fast speeds
        csz  = a2c2**2 - (2 * az * c)**2
        utils.filter.SQRTnegFilter(csz)
        
        cfz = np.sqrt(a2c2 + np.sqrt(csz))/ np.sqrt(2)
        
        csz = a2c2 - np.sqrt(csz)
        utils.filter.SQRTnegFilter(csz)
        
        csz = np.sqrt(csz) / np.sqrt(2)
        
        del(a2c2)
        
        mem = mem + (sys.getsizeof(csx)+sys.getsizeof(csy)+sys.getsizeof(csz) +
                     sys.getsizeof(cfx)+sys.getsizeof(cfy)+sys.getsizeof(cfz) +
                     sys.getsizeof(c)) / const.gig
        print("rank = ", ct.rank, " Memory used after speeds = ", mem, " Gb\n")
        
        # Save the characteristic speeds if necessary
        if ct.SpCond:
            hdfVel = h5py.File(dirDict["speed"] + "Speed_" + filename[0:-4] + ".h5", 'w')
            hdfVel.create_dataset('ax' , data=np.array(ax ,dtype='float64'), compression='gzip', compression_opts=9)
            hdfVel.create_dataset('ay' , data=np.array(ay ,dtype='float64'), compression='gzip', compression_opts=9)
            hdfVel.create_dataset('az' , data=np.array(az ,dtype='float64'), compression='gzip', compression_opts=9)
            hdfVel.create_dataset('csx', data=np.array(csx,dtype='float64'), compression='gzip', compression_opts=9)
            hdfVel.create_dataset('csy', data=np.array(csy,dtype='float64'), compression='gzip', compression_opts=9)
            hdfVel.create_dataset('csz', data=np.array(csz,dtype='float64'), compression='gzip', compression_opts=9)
            hdfVel.create_dataset('cfx', data=np.array(cfx,dtype='float64'), compression='gzip', compression_opts=9)
            hdfVel.create_dataset('cfy', data=np.array(cfy,dtype='float64'), compression='gzip', compression_opts=9)
            hdfVel.create_dataset('cfz', data=np.array(cfz,dtype='float64'), compression='gzip', compression_opts=9)
            hdfVel.close()
        
        
        #**********************************************************************************************************
        if ct.rank == ct.mainrank: print("Computing the Eigenenergy time derivatives...\n")
        
        # DivB: m = 1
        divx = -(((ax * BxDx * vx) * sqmurho) / mu0)
        divy = -(((ay * ByDy * vy) * sqmurho) / mu0)
        divz = -(((az * BzDz * vz) * sqmurho) / mu0)
        
        # Save the divB eigenenergy time derivatives
        hdfnew = h5py.File(dirDict["eq6"]+"EigenenergyDDT_"+filename[0:-4]+'.h5', 'w')
        hdfnew.create_dataset('eq6_m1_x', data=np.array(divx,dtype='float64'), compression='gzip', compression_opts=9)
        hdfnew.create_dataset('eq6_m1_y', data=np.array(divy,dtype='float64'), compression='gzip', compression_opts=9)
        hdfnew.create_dataset('eq6_m1_z', data=np.array(divz,dtype='float64'), compression='gzip', compression_opts=9)
        
        del(divx,
        divy,
        divz)
        
        # If requested, compute and save the divB error term according to Paper II if requested by user
        if ct.ErCond:
            div1 = ((ax * vx + ay * vy + az * vz) * (BxDx + ByDy + BzDz) * sqmurho) / mu0
            hdfnew.create_dataset('eq6_m1_err', data=np.array(div1,dtype='float64'), compression='gzip', compression_opts=9)
            del(div1)
        
        # No longer need BqDq; update the used memory
        mem  = mem - (sys.getsizeof(BxDx)+sys.getsizeof(ByDy)+sys.getsizeof(BzDz)) / const.gig
        del(BxDx, 
            ByDy,
            BzDz)
        
        # Save the time
        hdfnew.attrs['time'] = time
        
        # Ent and gravity PA modes: m = 2 and 9: currently the code can only handle constant gravity along -z 
        pDx  = methods.diff.partial(p,0,readObj.xc)
        pDy  = methods.diff.partial(p,1,readObj.yc)
        pDz  = methods.diff.partial(p,2,readObj.zc)
        # We no longer need p
        del(p)
        
        rhoDx = methods.diff.partial(rho,0,readObj.xc)
        rhoDy = methods.diff.partial(rho,1,readObj.yc)
        rhoDz = methods.diff.partial(rho,2,readObj.zc)
        
        # Update the used memory
        mem = mem + (sys.getsizeof(pDx)   + sys.getsizeof(pDy)   + sys.getsizeof(pDz) +
                     sys.getsizeof(rhoDx) + sys.getsizeof(rhoDy) + sys.getsizeof(rhoDz)) / const.gig
        print("rank = ", ct.rank, " Memory used after pDq and rhoDq just before Ent = ", mem, " Gb\n")
        
        Entx = vsq * (vx * (pDx - c**2 * rhoDx)) / 2 / c**2
        Enty = vsq * (vy * (pDy - c**2 * rhoDy)) / 2 / c**2
        Entz = vsq * (vz * (pDz - c**2 * rhoDz)) / 2 / c**2
        hdfnew.create_dataset('eq6_m9_x', data=np.array([0],dtype='float64'), compression='gzip', compression_opts=9)
        hdfnew.create_dataset('eq6_m9_y', data=np.array([0],dtype='float64'), compression='gzip', compression_opts=9)
        if g == 0:
            hdfnew.create_dataset('eq6_m9_z', data=np.array([0],dtype='float64'), compression='gzip', compression_opts=9)
        else:
            Entx = Entx + g*vx*readObj.zc*(pDx/c**2 - rhoDx)
            Enty = Enty + g*vy*readObj.zc*(pDy/c**2 - rhoDy)
            Entz = Entz + (g*pDz*vz*readObj.zc)/c**2 - g*vz*readObj.zc*rhoDz
            
            hdfnew.create_dataset('eq6_m9_z', data=np.array(- rho * g * vz,dtype='float64'), compression='gzip', compression_opts=9)
        
        #**********************************************************************************************************
        hdfnew.create_dataset('eq6_m2_x', data=np.array(Entx,dtype='float64'), compression='gzip', compression_opts=9)
        hdfnew.create_dataset('eq6_m2_y', data=np.array(Enty,dtype='float64'), compression='gzip', compression_opts=9)
        hdfnew.create_dataset('eq6_m2_z', data=np.array(Entz,dtype='float64'), compression='gzip', compression_opts=9)
        
        del(Entx,
            Enty,
            Entz)
        
        # IMPORTANT: We want to check whether different parts of the domain contain a wave-field or not
        # This will help identify possible nans in the outputs
        maskWaveField = vsq < utils.filter.smallestWaveFieldAllowed
        mem           = mem + sys.getsizeof(maskWaveField) / const.gig
        hdfnew.create_dataset('maskWave', data=np.array(maskWaveField,dtype='bool'), compression='gzip', compression_opts=9)
        waveCond = np.any(maskWaveField)
        
        # Compute the velocity derivatives
        vxDx = methods.diff.partial(vx,0,readObj.xc)
        vxDy = methods.diff.partial(vx,1,readObj.yc)
        vxDz = methods.diff.partial(vx,2,readObj.zc)
        vyDx = methods.diff.partial(vy,0,readObj.xc)
        vyDy = methods.diff.partial(vy,1,readObj.yc)
        vyDz = methods.diff.partial(vy,2,readObj.zc)
        vzDx = methods.diff.partial(vz,0,readObj.xc)
        vzDy = methods.diff.partial(vz,1,readObj.yc)
        vzDz = methods.diff.partial(vz,2,readObj.zc)
        
        mem = mem + (sys.getsizeof(vxDx)+sys.getsizeof(vxDy)+sys.getsizeof(vxDz) +
                    sys.getsizeof(vyDx)+sys.getsizeof(vyDy)+sys.getsizeof(vyDz)  +
                    sys.getsizeof(vzDx)+sys.getsizeof(vzDy)+sys.getsizeof(vzDz)) / const.gig
        
        print("rank = ", ct.rank, " Memory used before computing the Alfven eigenenergy time derivatives = ", mem, " Gb\n")
        
        sx = np.sign(ax)
        
        alfx1 = -0.5*((az*vy - ay*vz)*rho*(vx - np.abs(ax))*((az*vyDx - ay*vzDx)*sqmurho + (az*ByDx - ay*BzDx)*sx))/((ay**2 + az**2)*sqmurho)
        if waveCond: alfx1[maskWaveField] = 0.
        hdfnew.create_dataset('eq6_m3_x', data=np.array(alfx1,dtype='float64'), compression='gzip', compression_opts=9)
        del(alfx1)
            
        alfx2 = -0.5*((az*vy - ay*vz)*rho*(vx + np.abs(ax))*((az*vyDx - ay*vzDx)*sqmurho + (-(az*ByDx) + ay*BzDx)*sx))/((ay**2 + az**2)*sqmurho)
        if waveCond: alfx2[maskWaveField] = 0.
        hdfnew.create_dataset('eq6_m4_x', data=np.array(alfx2,dtype='float64'), compression='gzip', compression_opts=9)
        del(alfx2)
        
        print("rank = ", ct.rank, " Memory used after Alfven = ", mem, " Gb\n")
        
        # Compute the alpha variables in the x-direction
        alphasx = (cfx**2 - c**2)/(cfx**2 - csx**2)
        alphafx = (c**2 - csx**2)/(cfx**2 - csx**2)
        
        # Check if the alpha variables have negative elements (for an ideal plasma, they should not!)
        mask = alphasx < 0
        if len(alphasx[mask]) > 0:
            if -np.min(alphasx[mask]) > 10**-10:
                print("•••••••••••••••• Warning! alphasx has negative elements bigger than 10**-10. We had np.min(alphasx[mask]) = ",np.min(alphasx[mask]))
            
            print("This should not be happening for an ideal plasma. Check your simulation. Setting the negative elements of alphasx to zero!\n")
            alphasx[mask] = 0

        mask = alphafx < 0
        if len(alphafx[mask]) > 0:
            if -np.min(alphafx[mask]) > 10**-10:
                print("•••••••••••••••• Warning! alphafx has negative elements bigger than 10**-10. We had np.min(alphafx[mask]) = ",np.min(alphafx[mask]))
            
            print("This should not be happening for an ideal plasma. Check your simulation. Setting the negative elements of alphafx to zero!\n")
            alphafx[mask] = 0


        alphasx = np.sqrt(alphasx)
        alphafx = np.sqrt(alphafx)
        
        
        # Slowx1 (m = 5 and 6)
        slowx1 = (((csx - vx)*(np.sqrt(ay**2 + az**2)*c*alphafx + csx*vx*alphasx - ((vx**2 + vy**2 + vz**2)*alphasx)/2. - (c**2*alphasx)/(-1 + gamma) +
                (cfx*(ay*vy + az*vz)*alphafx*sx)/np.sqrt(ay**2 + az**2))*(sqmurho*(-(pDx*alphasx) + csx*vxDx*alphasx*rho)                            + 
                (ay*alphafx*rho*(ByDx*c + cfx*vyDx*sqmurho*sx))/np.sqrt(ay**2 + az**2)                                                               + 
                (az*alphafx*rho*(BzDx*c + cfx*vzDx*sqmurho*sx))/np.sqrt(ay**2 + az**2)))/(2.*c**2*sqmurho))
        
        slowx2 = (-0.5*((csx + vx)*(np.sqrt(ay**2 + az**2)*c*alphafx - ((2*csx*vx + vx**2 + vy**2 + vz**2)*alphasx)/2. - (c**2*alphasx)/(-1 + gamma)  - 
                                    (cfx*(ay*vy + az*vz)*alphafx*sx)/np.sqrt(ay**2 + az**2))*(-(alphasx*sqmurho*(pDx + csx*vxDx*rho))                 + 
                                    (alphafx*rho*((ay*ByDx + az*BzDx)*c - cfx*(ay*vyDx + az*vzDx)*sqmurho*sx))/np.sqrt(ay**2 + az**2)))/(c**2*sqmurho))
        
        # Additional terms due to gravity
        if g == 0:
            pass
        else:
            slowx1 = slowx1 + (-0.5*(g*(csx - vx)*readObj.zc*alphasx*(ay*ByDx*c*alphafx*rho + az*BzDx*c*alphafx*rho - np.sqrt(ay**2 + az**2)*pDx*alphasx*sqmurho + 
                    np.sqrt(ay**2 + az**2)*csx*vxDx*alphasx*rho*sqmurho + cfx*(ay*vyDx + az*vzDx)*alphafx*rho*sqmurho*sx))/(np.sqrt(ay**2 + az**2)*c**2*sqmurho))
            
            slowx2 = slowx2 + (-0.5*(g*(csx + vx)*readObj.zc*alphasx*(-(ay*ByDx*c*alphafx*rho) - az*BzDx*c*alphafx*rho + np.sqrt(ay**2 + az**2)*pDx*alphasx*sqmurho + 
                    np.sqrt(ay**2 + az**2)*csx*vxDx*alphasx*rho*sqmurho + cfx*(ay*vyDx + az*vzDx)*alphafx*rho*sqmurho*sx))/(np.sqrt(ay**2 + az**2)*c**2*sqmurho))
        
        mem = mem + (sys.getsizeof(slowx1) + sys.getsizeof(slowx2)) / const.gig
        print("rank = ", ct.rank, " Memory used after computing the x-directed slow mode terms = ", mem, " Gb\n")
        
        if waveCond: 
            slowx1[maskWaveField] = 0.
            slowx2[maskWaveField] = 0.
        
        if set.mode == "XYZUpDownSeparated":
            hdfnew.create_dataset('eq6_m5_x', data=np.array(slowx1,dtype='float64'), compression='gzip', compression_opts=9)
            hdfnew.create_dataset('eq6_m6_x', data=np.array(slowx2,dtype='float64'), compression='gzip', compression_opts=9)
        elif set.mode == "UpDownCombined":
            hdfnew.create_dataset('eq6_m5and6_x', data=np.array(slowx1 + slowx2,dtype='float64'), compression='gzip', compression_opts=9)
        
        del(slowx1, 
            slowx2)
        
        
        # Fastx (m = 7 and 8)
        fastx1 = (((cfx - vx)*(cfx*vx*alphafx - ((vx**2 + vy**2 + vz**2)*alphafx)/2. - np.sqrt(ay**2 + az**2)*c*alphasx - (c**2*alphafx)/(-1 + gamma)            - 
                (csx*(ay*vy + az*vz)*alphasx*sx)/np.sqrt(ay**2 + az**2))*(sqmurho*(-(pDx*alphafx) + cfx*vxDx*alphafx*rho) - (alphasx*rho*((ay*ByDx + az*BzDx)*c + 
                csx*(ay*vyDx + az*vzDx)*sqmurho*sx))/np.sqrt(ay**2 + az**2)))/(2.*c**2*sqmurho))
        
        fastx2 = (-0.25*((cfx + vx)*(2*np.sqrt(ay**2 + az**2)*c**2*alphafx + np.sqrt(ay**2 + az**2)*(2*cfx*vx + vx**2 + vy**2 + vz**2)*alphafx*(-1 + gamma)          + 
                2*(ay**2 + az**2)*c*alphasx*(-1 + gamma) - 2*csx*(ay*vy + az*vz)*alphasx*(-1 + gamma)*sx)*(ay*ByDx*c*alphasx*rho + az*BzDx*c*alphasx*rho            + 
                np.sqrt(ay**2 + az**2)*pDx*alphafx*sqmurho + np.sqrt(ay**2 + az**2)*cfx*vxDx*alphafx*rho*sqmurho - csx*(ay*vyDx + az*vzDx)*alphasx*rho*sqmurho*sx)) /
                ((ay**2 + az**2)*c**2*(-1 + gamma)*sqmurho))
        
        if g == 0:
            pass
        else:
            fastx1 = fastx1 + (-0.5*(g*(cfx - vx)*readObj.zc*alphafx*(-(ay*ByDx*c*alphasx*rho) - az*BzDx*c*alphasx*rho - np.sqrt(ay**2 + az**2)*pDx*alphafx*sqmurho + 
                    np.sqrt(ay**2 + az**2)*cfx*vxDx*alphafx*rho*sqmurho - csx*(ay*vyDx + az*vzDx)*alphasx*rho*sqmurho*sx))/(np.sqrt(ay**2 + az**2)*c**2*sqmurho))
            
            fastx2 = fastx2 + (-0.5*(g*(cfx + vx)*readObj.zc*alphafx*(ay*ByDx*c*alphasx*rho + az*BzDx*c*alphasx*rho + np.sqrt(ay**2 + az**2)*pDx*alphafx*sqmurho + 
                    np.sqrt(ay**2 + az**2)*cfx*vxDx*alphafx*rho*sqmurho - csx*(ay*vyDx + az*vzDx)*alphasx*rho*sqmurho*sx))/(np.sqrt(ay**2 + az**2)*c**2*sqmurho))
        
        del(alphasx,
            alphafx,
            sx)
        
        if waveCond: 
            fastx1[maskWaveField] = 0.
            fastx2[maskWaveField] = 0.
        
        if set.mode == "XYZUpDownSeparated":
            hdfnew.create_dataset('eq6_m7_x', data=np.array(fastx1,dtype='float64'), compression='gzip', compression_opts=9)
            hdfnew.create_dataset('eq6_m8_x', data=np.array(fastx2,dtype='float64'), compression='gzip', compression_opts=9)
        elif set.mode == "UpDownCombined":
            hdfnew.create_dataset('eq6_m7and8_x', data=np.array(fastx1 + fastx2,dtype='float64'), compression='gzip', compression_opts=9)
        
        del(fastx1, 
            fastx2)
        
        # If the allocated memory can handle the program up to this point then the rest will be fine
        mem = mem - sys.getsizeof(pDx) / const.gig
        del(pDx)
        print("rank = ", ct.rank, " Last memory check: memory used = ", mem, " Gb\n")
        
        alphasy = (cfy**2 - c**2)/(cfy**2 - csy**2)
        alphafy = (c**2 - csy**2)/(cfy**2 - csy**2)
        sy      = np.sign(ay)
        
        mask = alphasy < 0
        if len(alphasy[mask]) > 0:
            if -np.min(alphasy[mask]) > 10**-10:
                print("•••••••••••••••• Warning! alphasy has negative elements bigger than 10**-10. We had np.min(alphasy[mask]) = ",np.min(alphasy[mask]))
            
            print("This should not be happening for an ideal plasma. Check your simulation. Setting the negative elements of alphasy to zero!\n")
            alphasy[mask] = 0

        mask = alphafy < 0
        if len(alphafy[mask]) > 0:
            if -np.min(alphafy[mask]) > 10**-10:
                print("•••••••••••••••• Warning! alphafy has negative elements bigger than 10**-10. We had np.min(alphafy[mask]) = ",np.min(alphafy[mask]))
            
            print("This should not be happening for an ideal plasma. Check your simulation. Setting the negative elements of alphafy to zero!\n")
            alphafy[mask] = 0


        alphasy = np.sqrt(alphasy)
        alphafy = np.sqrt(alphafy)
        
        alfy1 = -0.5*((az*vx - ax*vz)*rho*(vy - np.abs(ay))*((az*vxDy - ax*vzDy)*sqmurho + (az*BxDy - ax*BzDy)*sy))/((ax**2 + az**2)*sqmurho)
        if waveCond: alfy1[maskWaveField] = 0.
        hdfnew.create_dataset('eq6_m3_y', data=np.array(alfy1,dtype='float64'), compression='gzip', compression_opts=9)
        del(alfy1)
        
        alfy2 = -0.5*((az*vx - ax*vz)*rho*(vy + np.abs(ay))*((az*vxDy - ax*vzDy)*sqmurho + (-(az*BxDy) + ax*BzDy)*sy))/((ax**2 + az**2)*sqmurho)
        if waveCond: alfy2[maskWaveField] = 0.
        hdfnew.create_dataset('eq6_m4_y', data=np.array(alfy2,dtype='float64'), compression='gzip', compression_opts=9)
        del(alfy2)
        
        # Slowy
        slowy1 = (((csy - vy)*(np.sqrt(ax**2 + az**2)*c*alphafy + csy*vy*alphasy - ((vx**2 + vy**2 + vz**2)*alphasy)/2. - (c**2*alphasy)/(-1 + gamma) + 
                (cfy*(ax*vx + az*vz)*alphafy*sy)/np.sqrt(ax**2 + az**2))*(sqmurho*(-(pDy*alphasy) + csy*vyDy*alphasy*rho)                            + 
                (ax*alphafy*rho*(BxDy*c + cfy*vxDy*sqmurho*sy))/np.sqrt(ax**2 + az**2)                                                               + 
                (az*alphafy*rho*(BzDy*c + cfy*vzDy*sqmurho*sy))/np.sqrt(ax**2 + az**2)))/(2.*c**2*sqmurho))
        
        slowy2 = (-0.5*((csy + vy)*(np.sqrt(ax**2 + az**2)*c*alphafy - ((vx**2 + 2*csy*vy + vy**2 + vz**2)*alphasy)/2. - (c**2*alphasy)/(-1 + gamma)      - 
                (cfy*(ax*vx + az*vz)*alphafy*sy)/np.sqrt(ax**2 + az**2))*(-(alphasy*sqmurho*(pDy + csy*vyDy*rho)) + (alphafy*rho*((ax*BxDy + az*BzDy)*c  - 
                cfy*(ax*vxDy + az*vzDy)*sqmurho*sy))/np.sqrt(ax**2 + az**2)))/(c**2*sqmurho))
            
        if g == 0:
            pass
        else:
            slowy1 = slowy1 + (-0.5*(g*(csy - vy)*readObj.zc*alphasy*(ax*BxDy*c*alphafy*rho + az*BzDy*c*alphafy*rho - np.sqrt(ax**2 + az**2)*pDy*alphasy*sqmurho + 
                    np.sqrt(ax**2 + az**2)*csy*vyDy*alphasy*rho*sqmurho + cfy*(ax*vxDy + az*vzDy)*alphafy*rho*sqmurho*sy))/(np.sqrt(ax**2 + az**2)*c**2*sqmurho))
            
            slowy2 = slowy2 + (-0.5*(g*(csy + vy)*readObj.zc*alphasy*(-(ax*BxDy*c*alphafy*rho) - az*BzDy*c*alphafy*rho + np.sqrt(ax**2 + az**2)*pDy*alphasy*sqmurho + 
                    np.sqrt(ax**2 + az**2)*csy*vyDy*alphasy*rho*sqmurho + cfy*(ax*vxDy + az*vzDy)*alphafy*rho*sqmurho*sy))/(np.sqrt(ax**2 + az**2)*c**2*sqmurho))
        
        if waveCond:
            slowy1[maskWaveField] = 0.
            slowy2[maskWaveField] = 0.
        
        if set.mode == "XYZUpDownSeparated":
            hdfnew.create_dataset('eq6_m5_y', data=np.array(slowy1,dtype='float64'), compression='gzip', compression_opts=9)
            hdfnew.create_dataset('eq6_m6_y', data=np.array(slowy2,dtype='float64'), compression='gzip', compression_opts=9)
        elif set.mode == "UpDownCombined":
            hdfnew.create_dataset('eq6_m5and6_y', data=np.array(slowy1 + slowy2,dtype='float64'), compression='gzip', compression_opts=9)
        
        del(slowy1, 
            slowy2)

        # Fasty
        fasty1 = (((cfy - vy)*(cfy*vy*alphafy - ((vx**2 + vy**2 + vz**2)*alphafy)/2. - np.sqrt(ax**2 + az**2)*c*alphasy - (c**2*alphafy)/(-1 + gamma) - 
                (csy*(ax*vx + az*vz)*alphasy*sy)/np.sqrt(ax**2 + az**2))*(sqmurho*(-(pDy*alphafy) + cfy*vyDy*alphafy*rho) - 
                (alphasy*rho*((ax*BxDy + az*BzDy)*c + csy*(ax*vxDy + az*vzDy)*sqmurho*sy))/np.sqrt(ax**2 + az**2)))/(2.*c**2*sqmurho))
        
        fasty2 = (-0.25*((cfy + vy)*(2*np.sqrt(ax**2 + az**2)*c**2*alphafy + np.sqrt(ax**2 + az**2)*(vx**2 + 2*cfy*vy + vy**2 + vz**2)*alphafy*(-1 + gamma)          + 
                2*(ax**2 + az**2)*c*alphasy*(-1 + gamma) - 2*csy*(ax*vx + az*vz)*alphasy*(-1 + gamma)*sy)*(ax*BxDy*c*alphasy*rho + az*BzDy*c*alphasy*rho            + 
                np.sqrt(ax**2 + az**2)*pDy*alphafy*sqmurho + np.sqrt(ax**2 + az**2)*cfy*vyDy*alphafy*rho*sqmurho - csy*(ax*vxDy + az*vzDy)*alphasy*rho*sqmurho*sy)) /
                ((ax**2 + az**2)*c**2*(-1 + gamma)*sqmurho))
                
        if g == 0:
            pass
        else:
            fasty1 = fasty1 + (-0.5*(g*(cfy - vy)*readObj.zc*alphafy*(-(ax*BxDy*c*alphasy*rho) - az*BzDy*c*alphasy*rho - np.sqrt(ax**2 + az**2)*pDy*alphafy*sqmurho + 
                    np.sqrt(ax**2 + az**2)*cfy*vyDy*alphafy*rho*sqmurho - csy*(ax*vxDy + az*vzDy)*alphasy*rho*sqmurho*sy))/(np.sqrt(ax**2 + az**2)*c**2*sqmurho))
            
            fasty2 = fasty2 + (-0.5*(g*(cfy + vy)*readObj.zc*alphafy*(ax*BxDy*c*alphasy*rho + az*BzDy*c*alphasy*rho + np.sqrt(ax**2 + az**2)*pDy*alphafy*sqmurho + 
                    np.sqrt(ax**2 + az**2)*cfy*vyDy*alphafy*rho*sqmurho - csy*(ax*vxDy + az*vzDy)*alphasy*rho*sqmurho*sy))/(np.sqrt(ax**2 + az**2)*c**2*sqmurho))
        
        del(alphasy,
            alphafy,
            sy)
        
        if waveCond: 
            fasty1[maskWaveField] = 0.
            fasty2[maskWaveField] = 0.
        
        if set.mode == "XYZUpDownSeparated":
            hdfnew.create_dataset('eq6_m7_y', data=np.array(fasty1,dtype='float64'), compression='gzip', compression_opts=9)
            hdfnew.create_dataset('eq6_m8_y', data=np.array(fasty2,dtype='float64'), compression='gzip', compression_opts=9)
        elif set.mode == "UpDownCombined":
            hdfnew.create_dataset('eq6_m7and8_y', data=np.array(fasty1 + fasty2,dtype='float64'), compression='gzip', compression_opts=9)
        
        del(fasty1, 
            fasty2)
        
        
        alphasz = (cfz**2 - c**2)/(cfz**2 - csz**2)
        alphafz = (c**2 - csz**2)/(cfz**2 - csz**2)
        sz      = np.sign(az)
        
        mask = alphasz < 0
        if len(alphasz[mask]) > 0:
            if -np.min(alphasz[mask]) > 10**-10:
                print("•••••••••••••••• Warning! alphasz has negative elements bigger than 10**-10. We had np.min(alphasz[mask]) = ",np.min(alphasz[mask]))
            
            print("This should not be happening for an ideal plasma. Check your simulation. Setting the negative elements of alphasz to zero!\n")
            alphasz[mask] = 0

        mask = alphafz < 0
        if len(alphafz[mask]) > 0:
            if -np.min(alphafz[mask]) > 10**-10:
                print("•••••••••••••••• Warning! alphafz has negative elements bigger than 10**-10. We had np.min(alphafz[mask]) = ",np.min(alphafz[mask]))
            
            print("This should not be happening for an ideal plasma. Check your simulation. Setting the negative elements of alphafz to zero!\n")
            alphafz[mask] = 0


        alphasz = np.sqrt(alphasz)
        alphafz = np.sqrt(alphafz)
        
        alfz1 = -0.5*((ay*vx - ax*vy)*rho*(vz - np.abs(az))*((ay*vxDz - ax*vyDz)*sqmurho + (ay*BxDz - ax*ByDz)*sz))/((ax**2 + ay**2)*sqmurho)
        if waveCond: alfz1[maskWaveField] = 0.
        hdfnew.create_dataset('eq6_m3_z', data=np.array(alfz1,dtype='float64'), compression='gzip', compression_opts=9)
        del(alfz1)
        
        alfz2 = -0.5*((ay*vx - ax*vy)*rho*(vz + np.abs(az))*((ay*vxDz - ax*vyDz)*sqmurho + (-(ay*BxDz) + ax*ByDz)*sz))/((ax**2 + ay**2)*sqmurho)
        if waveCond: alfz2[maskWaveField] = 0.
        hdfnew.create_dataset('eq6_m4_z', data=np.array(alfz2,dtype='float64'), compression='gzip', compression_opts=9)
        del(alfz2)
        
        # Slowz
        slowz1 = (((csz - vz)*(np.sqrt(ax**2 + ay**2)*c*alphafz + csz*vz*alphasz - ((vx**2 + vy**2 + vz**2)*alphasz)/2. - (c**2*alphasz)/(-1 + gamma) + 
                (cfz*(ax*vx + ay*vy)*alphafz*sz)/np.sqrt(ax**2 + ay**2))*(sqmurho*(-(pDz*alphasz) + csz*vzDz*alphasz*rho)                            + 
                (ax*alphafz*rho*(BxDz*c + cfz*vxDz*sqmurho*sz))/np.sqrt(ax**2 + ay**2) + (ay*alphafz*rho*(ByDz*c + cfz*vyDz*sqmurho*sz))             /
                np.sqrt(ax**2 + ay**2)))/(2.*c**2*sqmurho))
        
        slowz2 = (-0.5*((csz + vz)*(np.sqrt(ax**2 + ay**2)*c*alphafz - ((vx**2 + vy**2 + 2*csz*vz + vz**2)*alphasz)/2. - (c**2*alphasz)/(-1 + gamma)     - 
                (cfz*(ax*vx + ay*vy)*alphafz*sz)/np.sqrt(ax**2 + ay**2))*(-(alphasz*sqmurho*(pDz + csz*vzDz*rho)) + (alphafz*rho*((ax*BxDz + ay*ByDz)*c - 
                cfz*(ax*vxDz + ay*vyDz)*sqmurho*sz))/np.sqrt(ax**2 + ay**2)))/(c**2*sqmurho))
        
        if g == 0:
            pass
        else:
            slowz1 = slowz1 + (-0.5*(g*(csz - vz)*readObj.zc*alphasz*(ax*BxDz*c*alphafz*rho + ay*ByDz*c*alphafz*rho - np.sqrt(ax**2 + ay**2)*pDz*alphasz*sqmurho + 
                    np.sqrt(ax**2 + ay**2)*csz*vzDz*alphasz*rho*sqmurho + cfz*(ax*vxDz + ay*vyDz)*alphafz*rho*sqmurho*sz))/(np.sqrt(ax**2 + ay**2)*c**2*sqmurho))
            
            slowz2 = slowz2 + (-0.5*(g*(csz + vz)*readObj.zc*alphasz*(-(ax*BxDz*c*alphafz*rho) - ay*ByDz*c*alphafz*rho + np.sqrt(ax**2 + ay**2)*pDz*alphasz*sqmurho + 
                    np.sqrt(ax**2 + ay**2)*csz*vzDz*alphasz*rho*sqmurho + cfz*(ax*vxDz + ay*vyDz)*alphafz*rho*sqmurho*sz))/(np.sqrt(ax**2 + ay**2)*c**2*sqmurho))
        
        if waveCond: 
            slowz1[maskWaveField] = 0.
            slowz2[maskWaveField] = 0.
        
        if set.mode == "XYZUpDownSeparated":
            hdfnew.create_dataset('eq6_m5_z', data=np.array(slowz1,dtype='float64'), compression='gzip', compression_opts=9)
            hdfnew.create_dataset('eq6_m6_z', data=np.array(slowz2,dtype='float64'), compression='gzip', compression_opts=9)
        elif set.mode == "UpDownCombined":
            hdfnew.create_dataset('eq6_m5and6_z', data=np.array(slowz1 + slowz2,dtype='float64'), compression='gzip', compression_opts=9)
        
        del(slowz1, 
            slowz2)
        
        # Fastz
        fastz1 = (((cfz - vz)*(cfz*vz*alphafz - ((vx**2 + vy**2 + vz**2)*alphafz)/2. - np.sqrt(ax**2 + ay**2)*c*alphasz - (c**2*alphafz)/(-1 + gamma)            - 
                (csz*(ax*vx + ay*vy)*alphasz*sz)/np.sqrt(ax**2 + ay**2))*(sqmurho*(-(pDz*alphafz) + cfz*vzDz*alphafz*rho) - (alphasz*rho*((ax*BxDz + ay*ByDz)*c + 
                csz*(ax*vxDz + ay*vyDz)*sqmurho*sz))/np.sqrt(ax**2 + ay**2)))/(2.*c**2*sqmurho))
        
        fastz2 = (-0.25*((cfz + vz)*(2*np.sqrt(ax**2 + ay**2)*c**2*alphafz + np.sqrt(ax**2 + ay**2)*(vx**2 + vy**2 + vz*(2*cfz + vz))*alphafz*(-1 + gamma) + 
                2*(ax**2 + ay**2)*c*alphasz*(-1 + gamma) - 2*csz*(ax*vx + ay*vy)*alphasz*(-1 + gamma)*sz)*(ax*BxDz*c*alphasz*rho + ay*ByDz*c*alphasz*rho  + 
                np.sqrt(ax**2 + ay**2)*pDz*alphafz*sqmurho + np.sqrt(ax**2 + ay**2)*cfz*vzDz*alphafz*rho*sqmurho                                          - 
                csz*(ax*vxDz + ay*vyDz)*alphasz*rho*sqmurho*sz))/((ax**2 + ay**2)*c**2*(-1 + gamma)*sqmurho))
                        
        if g == 0:
            pass
        else:
            fastz1 = fastz1 + (-0.5*(g*(cfz - vz)*readObj.zc*alphafz*(-(ax*BxDz*c*alphasz*rho) - ay*ByDz*c*alphasz*rho - np.sqrt(ax**2 + ay**2)*pDz*alphafz*sqmurho + 
                    np.sqrt(ax**2 + ay**2)*cfz*vzDz*alphafz*rho*sqmurho - csz*(ax*vxDz + ay*vyDz)*alphasz*rho*sqmurho*sz))/(np.sqrt(ax**2 + ay**2)*c**2*sqmurho))
            
            fastz2 = fastz2 + (-0.5*(g*(cfz + vz)*readObj.zc*alphafz*(ax*BxDz*c*alphasz*rho + ay*ByDz*c*alphasz*rho + np.sqrt(ax**2 + ay**2)*pDz*alphafz*sqmurho + 
                    np.sqrt(ax**2 + ay**2)*cfz*vzDz*alphafz*rho*sqmurho - csz*(ax*vxDz + ay*vyDz)*alphasz*rho*sqmurho*sz))/(np.sqrt(ax**2 + ay**2)*c**2*sqmurho))
        
        del(pDz    ,
            rhoDz  ,
            sz     ,
            alphasz,
            alphafz)
        
        if waveCond: 
            fastz1[maskWaveField] = 0.
            fastz2[maskWaveField] = 0.
        
        if set.mode == "XYZUpDownSeparated":
            hdfnew.create_dataset('eq6_m7_z', data=np.array(fastz1,dtype='float64'), compression='gzip', compression_opts=9)
            hdfnew.create_dataset('eq6_m8_z', data=np.array(fastz2,dtype='float64'), compression='gzip', compression_opts=9)
        elif set.mode == "UpDownCombined":
            hdfnew.create_dataset('eq6_m7and8_z', data=np.array(fastz1 + fastz2,dtype='float64'), compression='gzip', compression_opts=9)
        
        del(fastz1, 
            fastz2)
        
        hdfnew.close()
        
        # Free up the memory
        del(sqmurho, 
            vxDx   ,
            vxDy   ,
            vxDz   ,
            vyDx   ,
            vyDy   ,
            vyDz   ,
            vzDx   ,
            vzDy   ,
            vzDz   ,
            BxDy   ,
            BxDz   ,
            ByDx   ,
            ByDz   ,
            BzDx   ,
            BzDy   ,
            ax     ,
            ay     ,
            az     ,
            csx    ,
            cfx    ,
            csy    ,
            cfy    ,
            csz    ,
            cfz    ,
            maskWaveField)

    if ct.rank == ct.mainrank:
        print("Equations 6 computed and saved in " + dirDict["eq6"] + ".\n")