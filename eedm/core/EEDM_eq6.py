'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: Script to compute Equations 6 of Paper III (linked above)
'''

import numpy as np
import os
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
def run(dirDict, readObj=None):
    # Import gamma, g, and mu0 for ease of use
    gamma = float(set.gamma)   # plain floats keep the arrays in the chosen precision
    g     = float(set.g)
    mu0   = const.mu0
    
    # Initialize the reading of the simulation data (unless main() already did, for the run header)
    if readObj is None:
        readObj = io.reader.EEDM_reader()
        if ct.rank == ct.mainrank:
            for note in readObj.notes: print(note + "\n", flush=True)

    # If Equation 9 follows, check its slicing points now (on the cropped grid), rather than after all of Equation 6
    if ct.eq9Cond:
        slabIdx, slabLoc, notes = utils.get_energy_sliced.resolve_slicing_points(set.slicingPlane, set.slicingPnts, getattr(set, "slicingPntsType", None),
                                                                                  readObj.xc, readObj.yc, readObj.zc)
        if ct.rank == ct.mainrank:
            normal = utils.get_energy_sliced.NORMAL[set.slicingPlane]
            for note in notes: print(note)
            print("Equation 9 will use " + str(len(slabIdx)) + " " + set.slicingPlane + "-slab(s), at " +
                  ", ".join("%s = %g (index %d)" % (normal, loc, idx) for loc, idx in zip(slabLoc, slabIdx)) + "\n", flush=True)

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
    filenames = [os.path.basename(readObj.feed[i]) for i in range(slt,elt)]
    
    if ct.rank == ct.mainrank:
        print("Equation 6: computing the eigenpowers of " + str(readObj.nt) + " snapshot(s) on " + str(ct.size) + " rank(s)", flush=True)
    
    # Compute the eigenenergy time derivatives (Equations 6 of Paper III)
    for i in range(slt,elt):
        filename                       = filenames[i - slt]
        stem                           = os.path.splitext(filename)[0] # file name without its extension (.sdf, .cfd, .h5, ...)
        rho, vx, vy, vz, bx, by, bz, p = readObj.data(i)
        time                           = readObj.time
        
        print("  [rank %d] %s (t = %g)" % (ct.rank, filename, time), flush=True)
        
        # Compute vsq and save for later
        vsq = vx**2 + vy**2 + vz**2
        
        if ct.EnCond:
            # Save the kinetic, magnetic, internal, and gravitational energies
            hdfEtot =  h5py.File(os.path.join(dirDict["energy"], "Etot_" + stem + ".h5"), 'w')
            
            hdfEtot.create_dataset('Kin', data=np.array(utils.orientation.array_out(0.5 * rho * vsq)                   , dtype=ct.dtype), compression='gzip', compression_opts=9) 
            hdfEtot.create_dataset('Mag', data=np.array(utils.orientation.array_out(0.5 * (bx**2 + by**2 + bz**2) / mu0), dtype=ct.dtype), compression='gzip', compression_opts=9) 
            hdfEtot.create_dataset('Int', data=np.array(utils.orientation.array_out(p / (gamma - 1))                   , dtype=ct.dtype), compression='gzip', compression_opts=9) 
            if g == 0:
                hdfEtot.create_dataset('Grv', data=np.array([0]                 ,dtype=ct.dtype), compression='gzip', compression_opts=9) 
            else:
                hdfEtot.create_dataset('Grv', data=np.array(utils.orientation.array_out(rho * g * readObj.ez),dtype=ct.dtype), compression='gzip', compression_opts=9)
            
            hdfEtot.close()
        
        
        # Save the grid and parameters in a single file
        
        if i == 0:
            hdfParam =  h5py.File(os.path.join(dirDict["parent"], "grid_params.h5"), 'w')
            hdfParam.create_dataset('xc',     data=np.array(readObj.xc    ,dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdfParam.create_dataset('yc',     data=np.array(readObj.yc    ,dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdfParam.create_dataset('zc',     data=np.array(readObj.zc    ,dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdfParam.create_dataset('framec', data=np.array(readObj.framec,dtype='float64'), compression='gzip', compression_opts=9)
            if not np.any(readObj.xb == None):
                hdfParam.create_dataset('frameb', data=np.array(readObj.frameb,dtype='float64'), compression='gzip', compression_opts=9)
                hdfParam.create_dataset('xb',     data=np.array(readObj.xb    ,dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdfParam.create_dataset('yb',     data=np.array(readObj.yb    ,dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdfParam.create_dataset('zb',     data=np.array(readObj.zb    ,dtype=ct.dtype), compression='gzip', compression_opts=9)
            
            hdfParam.attrs['decomposition'] = ct.decomposition
            hdfParam.attrs['gamma'   ] = gamma
            hdfParam.attrs['mu0'     ] = mu0
            hdfParam.attrs['g']        = g
            hdfParam.attrs['gCond']    = g > 0
            hdfParam.attrs['waveFieldTolerance'] = ct.waveFieldTolerance
            hdfParam.attrs['verticalAxis']       = ct.verticalAxis
            hdfParam.attrs['invariantAxes']      = ','.join(ct.invariantAxes)
            hdfParam.close()
        
        # Let's roughly keep track of how much memory we're using
        mem = sys.getsizeof(readObj) / const.gig
        mem = mem + (sys.getsizeof(vx )+sys.getsizeof(vy)+sys.getsizeof(vz)+
                     sys.getsizeof(bx )+sys.getsizeof(by)+sys.getsizeof(bz)+
                     sys.getsizeof(rho)+sys.getsizeof(vsq)) / const.gig
        if ct.verbose: print("  [rank %d] estimated memory in use before the derivatives: %.3g GB" % (ct.rank, mem), flush=True)
        
        # Compute the field derivatives
        BxDx = methods.diff.partial(bx,0,readObj.ex)
        BxDy = methods.diff.partial(bx,1,readObj.ey)
        BxDz = methods.diff.partial(bx,2,readObj.ez)
        ByDx = methods.diff.partial(by,0,readObj.ex)
        ByDy = methods.diff.partial(by,1,readObj.ey)
        ByDz = methods.diff.partial(by,2,readObj.ez)
        BzDx = methods.diff.partial(bz,0,readObj.ex)
        BzDy = methods.diff.partial(bz,1,readObj.ey)
        BzDz = methods.diff.partial(bz,2,readObj.ez)
        
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
        if ct.verbose: print("  [rank %d] estimated memory in use before the characteristic speeds: %.3g GB" % (ct.rank, mem), flush=True)
        
        # No longer need B
        del(bx,
            by,
            bz)
        
        # If necessary, compute and save divB and the polytropic k, where dk/dt|_Lagrangian = 0
        if ct.DbCond or ct.PkCond: 
            hdfExtra =  h5py.File(os.path.join(dirDict["extra"], "Extras_" + stem + ".h5"), 'w')
        
        if ct.PkCond:
            poly_k                           = p / rho**gamma            
            hdfExtra.create_dataset('poly_k', data=np.array(utils.orientation.array_out(poly_k),dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdfExtra.attrs['maxAbsEntropy' ] = np.max( np.abs(poly_k))
            hdfExtra.attrs['meanAbsEntropy'] = np.mean(np.abs(poly_k))
            hdfExtra.attrs['stdAbsEntropy' ] = np.std( np.abs(poly_k))
            del(poly_k)
        
        if ct.DbCond:
            divB                          = BxDx + ByDy + BzDz
            hdfExtra.create_dataset('divB', data=np.array(utils.orientation.array_out(divB),dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdfExtra.attrs['maxAbsDivB' ] = np.max(np.abs(divB))
            hdfExtra.attrs['meanAbsDivB'] = np.mean(np.abs(divB))
            hdfExtra.attrs['stdAbsDivB' ] = np.std(np.abs(divB))
            del(divB)
        
        if ct.DbCond or ct.PkCond: 
            hdfExtra.close()
        
        
        # Compute the characteristic speeds
        
        # Sound speed
        c   = np.sqrt(gamma * p / rho)
        c2  = c**2
        
        # Fast and slow speeds along each axis (cancellation-free forms, see utils.filter)
        cfx, csx = utils.filter.slow_fast_speeds(ax**2, ay**2 + az**2, c2)
        cfy, csy = utils.filter.slow_fast_speeds(ay**2, ax**2 + az**2, c2)
        cfz, csz = utils.filter.slow_fast_speeds(az**2, ax**2 + ay**2, c2)
        
        mem = mem + (sys.getsizeof(csx)+sys.getsizeof(csy)+sys.getsizeof(csz) +
                     sys.getsizeof(cfx)+sys.getsizeof(cfy)+sys.getsizeof(cfz) +
                     sys.getsizeof(c)) / const.gig
        if ct.verbose: print("  [rank %d] estimated memory in use after the characteristic speeds: %.3g GB" % (ct.rank, mem), flush=True)
        
        # Save the characteristic speeds if necessary
        if ct.SpCond:
            hdfVel = h5py.File(os.path.join(dirDict["speed"], "Speed_" + stem + ".h5"), 'w')
            # EEDM axis q is saved under its simulation axis letter
            for q, (aq, csq, cfq) in zip('xyz', ((ax, csx, cfx), (ay, csy, cfy), (az, csz, cfz))):
                a_ = utils.orientation.sim_axis(q)
                hdfVel.create_dataset('a'  + a_, data=np.array(utils.orientation.array_out(aq) ,dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdfVel.create_dataset('cs' + a_, data=np.array(utils.orientation.array_out(csq),dtype=ct.dtype), compression='gzip', compression_opts=9)
                hdfVel.create_dataset('cf' + a_, data=np.array(utils.orientation.array_out(cfq),dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdfVel.close()
        
        
        #**********************************************************************************************************
        
        # DivB: m = 1
        divx = -(((ax * BxDx * vx) * sqmurho) / mu0)
        divy = -(((ay * ByDy * vy) * sqmurho) / mu0)
        divz = -(((az * BzDz * vz) * sqmurho) / mu0)
        
        # Every mode m along q goes to one output of the chosen decomposition (utils.decomposition): add() sums the
        # contributions of each output and writes it as soon as all of them are in (data = None stands for exactly zero)
        hdfnew         = h5py.File(os.path.join(dirDict["eq6"], "EigenenergyDDT_" + stem + ".h5"), 'w')
        owner, pending = utils.decomposition.owner()
        partial_sums   = {}
        
        def add(m, q, data):
            name = owner[(m, q)]
            if q in ct.invariantAxesE: data = None          # no derivatives along an invariant axis, so exactly zero
            if data is not None:
                partial_sums[name] = data if name not in partial_sums else partial_sums[name] + data
            pending[name] -= 1
            if pending[name] == 0:
                hdfnew.create_dataset('eq6_' + name, data=np.array(utils.orientation.array_out(partial_sums.pop(name, [0])),dtype=ct.dtype), compression='gzip', compression_opts=9)
        
        # Save the divB eigenenergy time derivatives
        add(1, 'x', divx)
        add(1, 'y', divy)
        add(1, 'z', divz)
        
        del(divx,
        divy,
        divz)
        
        # If requested, compute and save the divB error term according to Paper II if requested by user
        if ct.ErCond:
            div1 = ((ax * vx + ay * vy + az * vz) * (BxDx + ByDy + BzDz) * sqmurho) / mu0
            hdfnew.create_dataset('eq6_m1_err', data=np.array(utils.orientation.array_out(div1),dtype=ct.dtype), compression='gzip', compression_opts=9)
            del(div1)
        
        # No longer need BqDq; update the used memory
        mem  = mem - (sys.getsizeof(BxDx)+sys.getsizeof(ByDy)+sys.getsizeof(BzDz)) / const.gig
        del(BxDx, 
            ByDy,
            BzDz)
        
        # Save the time
        hdfnew.attrs['time'] = time
        
        # Ent and gravity PA modes: m = 2 and 9: currently the code can only handle constant gravity along -z 
        pDx  = methods.diff.partial(p,0,readObj.ex)
        pDy  = methods.diff.partial(p,1,readObj.ey)
        pDz  = methods.diff.partial(p,2,readObj.ez)
        # We no longer need p
        del(p)
        
        rhoDx = methods.diff.partial(rho,0,readObj.ex)
        rhoDy = methods.diff.partial(rho,1,readObj.ey)
        rhoDz = methods.diff.partial(rho,2,readObj.ez)
        
        # Update the used memory
        mem = mem + (sys.getsizeof(pDx)   + sys.getsizeof(pDy)   + sys.getsizeof(pDz) +
                     sys.getsizeof(rhoDx) + sys.getsizeof(rhoDy) + sys.getsizeof(rhoDz)) / const.gig
        if ct.verbose: print("  [rank %d] estimated memory in use before the entropy mode: %.3g GB" % (ct.rank, mem), flush=True)
        
        Entx = vsq * (vx * (pDx - c2 * rhoDx)) / 2 / c2
        Enty = vsq * (vy * (pDy - c2 * rhoDy)) / 2 / c2
        Entz = vsq * (vz * (pDz - c2 * rhoDz)) / 2 / c2
        if ct.decomposition == "MHD_modes_xyz":
            # m9 only exists along z; keep the x and y placeholders so that every m_q has a dataset
            hdfnew.create_dataset('eq6_m9_' + utils.orientation.sim_axis('x'), data=np.array([0],dtype=ct.dtype), compression='gzip', compression_opts=9)
            hdfnew.create_dataset('eq6_m9_' + utils.orientation.sim_axis('y'), data=np.array([0],dtype=ct.dtype), compression='gzip', compression_opts=9)
        if g == 0:
            add(9, 'z', None)
        else:
            Entx = Entx + g*vx*readObj.ez*(pDx/c2 - rhoDx)
            Enty = Enty + g*vy*readObj.ez*(pDy/c2 - rhoDy)
            Entz = Entz + (g*pDz*vz*readObj.ez)/c2 - g*vz*readObj.ez*rhoDz
            
            add(9, 'z', - rho * g * vz)
        
        #**********************************************************************************************************
        add(2, 'x', Entx)
        add(2, 'y', Enty)
        add(2, 'z', Entz)
        
        del(Entx,
            Enty,
            Entz)
        
        # IMPORTANT: We want to check whether different parts of the domain contain a wave-field or not
        # This will help identify possible nans in the outputs
        maskWaveField = vsq < ct.waveFieldTolerance**2 * (ax**2 + ay**2 + az**2 + c2)
        mem           = mem + sys.getsizeof(maskWaveField) / const.gig
        hdfnew.create_dataset('maskWave', data=np.array(utils.orientation.array_out(maskWaveField),dtype='bool'), compression='gzip', compression_opts=9)
        waveCond = np.any(maskWaveField)
        
        # Compute the velocity derivatives
        vxDx = methods.diff.partial(vx,0,readObj.ex)
        vxDy = methods.diff.partial(vx,1,readObj.ey)
        vxDz = methods.diff.partial(vx,2,readObj.ez)
        vyDx = methods.diff.partial(vy,0,readObj.ex)
        vyDy = methods.diff.partial(vy,1,readObj.ey)
        vyDz = methods.diff.partial(vy,2,readObj.ez)
        vzDx = methods.diff.partial(vz,0,readObj.ex)
        vzDy = methods.diff.partial(vz,1,readObj.ey)
        vzDz = methods.diff.partial(vz,2,readObj.ez)
        
        mem = mem + (sys.getsizeof(vxDx)+sys.getsizeof(vxDy)+sys.getsizeof(vxDz) +
                    sys.getsizeof(vyDx)+sys.getsizeof(vyDy)+sys.getsizeof(vyDz)  +
                    sys.getsizeof(vzDx)+sys.getsizeof(vzDy)+sys.getsizeof(vzDz)) / const.gig
        
        if ct.verbose: print("  [rank %d] estimated memory in use before the Alfven, slow, and fast modes: %.3g GB" % (ct.rank, mem), flush=True)
        
        # Alfven, slow, and fast eigenenergy time derivatives (m = 3..8): Equations 6c and 6d of Paper III,
        # evaluated for each direction q with the conventions at the degenerate points (see utils.filter):
        #   beta_{q' perp q} = 1/sqrt(2) where a_perp,q = 0 (Eq. 7), s_q = +1 where a_q = 0, and
        #   alpha_s = alpha_f = 1/sqrt(2) at the triple point (a_perp,q = 0 and |a_q| = c)
        del(rhoDx, rhoDy, rhoDz)
        
        # c^2/(gamma - 1) + v^2/2 + phi, with phi = g z
        H = c2 / (gamma - 1) + 0.5 * vsq
        if g != 0: H = H + g * readObj.ez
        
        vel   = (vx, vy, vz)
        alf   = (ax, ay, az)
        speed = ((csx, cfx), (csy, cfy), (csz, cfz))
        # d_q of the velocity components, the magnetic field components (d_q B_q is not needed), and the pressure
        dVel  = [(vxDx, vyDx, vzDx), (vxDy, vyDy, vzDy), (vxDz, vyDz, vzDz)]
        dMag  = [(None, ByDx, BzDx), (BxDy, None, BzDy), (BxDz, ByDz, None)]
        dPre  = [pDx, pDy, pDz]
        del(vxDx, vyDx, vzDx, vxDy, vyDy, vzDy, vxDz, vyDz, vzDz, ByDx, BzDx, BxDy, BzDy, BxDz, ByDz, pDx, pDy, pDz)
        
        def save(m, q, data, mask=maskWaveField, masked=waveCond):   # this snapshot's quiet-point mask, bound explicitly
            if masked: data[mask] = 0.
            add(m, q, data)
        
        for q, qn in enumerate('xyz'):
            if qn in ct.invariantAxesE:                   # 2D and 1D runs: no Alfven, slow, or fast contribution along an invariant axis
                for m in range(3, 9): add(m, qn, None)
                dVel[q] = dMag[q] = dPre[q] = None
                continue
            q1, q2 = (q + 1) % 3, (q + 2) % 3        # the two perpendicular axes, in cyclic order
            vq, aq = vel[q], alf[q]
            cs, cf = speed[q]
            dv, dB, dp = dVel[q], dMag[q], dPre[q]
            
            s                = utils.filter.sgn(aq)
            ap, beta1, beta2 = utils.filter.perp_basis(alf[q1], alf[q2])       # row q of B: (beta1, beta2) on axes (q1, q2)
            alphas, alphaf   = utils.filter.slow_fast_alphas(aq**2, alf[q1]**2 + alf[q2]**2, c2)
            
            # Alfven (Eq. 6c): the q-components of sqrt(mu0 rho) v x B_q, sqrt(mu0 rho) d_q v x B_q, and s_q d_q B x B_q
            vX  = sqmurho * (vel[q1]*beta2 - vel[q2]*beta1)
            dvX = sqmurho * (dv[q1]*beta2 - dv[q2]*beta1)
            dBX = s * (dB[q1]*beta2 - dB[q2]*beta1)
            save(3, qn, -vX * (dvX + dBX) * (vq - np.abs(aq)) / (2*mu0))
            save(4, qn, -vX * (dvX - dBX) * (vq + np.abs(aq)) / (2*mu0))
            del(vX, dvX, dBX)
            
            # Slow and fast (Eq. 6d), with the projections v.B_q, d_q v.B_q, d_q B.B_q, and B.B_q = sqrt(mu0 rho) a_perp
            vB  = vel[q1]*beta1 + vel[q2]*beta2
            dvB = dv[q1]*beta1  + dv[q2]*beta2
            dBB = dB[q1]*beta1  + dB[q2]*beta2
            BB  = sqmurho * ap
            for (m, S, cThis, cOther, alThis, alOther) in ((5, 1, cs, cf, alphas, alphaf), (7, -1, cf, cs, alphaf, alphas)):
                # each bracket of Eq. 6d = (sign-independent part M) +/- (part N multiplying the upper/lower sign)
                M1 = S*alOther*c*BB - sqmurho*alThis*H
                N1 = S*alOther*s*cOther*sqmurho*vB + sqmurho*alThis*cThis*vq
                M2 = S*alOther*c*dBB - sqmurho*alThis*dp/rho
                N2 = S*alOther*s*cOther*sqmurho*dvB + sqmurho*alThis*cThis*dv[q]
                rev = -(M1 + N1) * (M2 + N2) * (vq - cThis) / (2*mu0*c2)
                fwd = -(M1 - N1) * (M2 - N2) * (vq + cThis) / (2*mu0*c2)
                save(m    , qn, rev)
                save(m + 1, qn, fwd)
                del(M1, N1, M2, N2, rev, fwd)
            
            del(s, ap, beta1, beta2, alphas, alphaf, vB, dvB, dBB, BB, dv, dB, dp)
            dVel[q] = dMag[q] = dPre[q] = None               # free this direction's derivatives
        
        del(H, dVel, dMag, dPre)
        
        if partial_sums or any(pending.values()):
            raise RuntimeError("Internal error: the " + ct.decomposition + " outputs " + str(sorted(set(partial_sums) | {k for k, v in pending.items() if v})) +
                               " of " + filename + " were not completed. Please report this to raboonik@gmail.com.")
        hdfnew.close()
        
        # Free up the memory
        del(sqmurho, vel, alf, speed, ax, ay, az, csx, cfx, csy, cfy, csz, cfz, maskWaveField)

    ct.comm.barrier()   # report once every rank has finished its snapshots
    if ct.rank == ct.mainrank:
        print("Equation 6 done: " + str(readObj.nt) + " file(s) EigenenergyDDT_*.h5 in " + dirDict["eq6"] + "\n", flush=True)