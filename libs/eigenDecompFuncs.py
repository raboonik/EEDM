from libs import prerequisites as pre

'''
    Author: Axel Raboonik
    Email : raboonik@gmail.com
    
    Github: https://github.com/raboonik
    
    Article: https://iopscience.iop.org/article/10.3847/1538-4357/adc917
    
    Description: Functions to aid in reading simulation data and working out 
                 the subgrid associated with the cropping frame 
'''

def lare3dGetCellVals(data,quantity):
    '''
        Here we perform 2-point-per-direction interpolations of various raw quantities.
        This is the standard interpolation scheme used in Lare.
    '''
    if "v" in quantity:
        out = 0.125 * (data + pre.np.roll(data, -1, axis=0) + pre.np.roll(data, -1, axis=1) + pre.np.roll(data, -1, axis=(0,1)) + 
        pre.np.roll(data, -1, axis=2) + pre.np.roll(data, -1, axis=(0,2)) + pre.np.roll(data, -1, axis=(1,2)) + pre.np.roll(data, -1, axis=(0,1,2)))  
        out = out[0:-1,0:-1,0:-1]
    elif quantity == "bx":
        out = 0.5*(data + pre.np.roll(data,-1,axis=0))
        out = out[0:-1,:,:]    
    elif quantity == "by":
        out = 0.5*(data + pre.np.roll(data,-1,axis=1))
        out = out[:,0:-1,:]    
    elif quantity == "bz":
        out = 0.5*(data + pre.np.roll(data,-1,axis=2))
        out = out[:,:,0:-1]
    else:
        out = pre.np.nan
        print("Enter an appropriate quantitiy! Returning NaN.")
    
    return out

def SQRTnegFilter(dat):
    if pre.np.min(dat) < 0:
        mask = dat < 0
        if pre.np.abs(pre.np.max(dat[mask]) - pre.np.min(dat[mask])) < 1.e-9:
            dat[mask] = 0.
    else:
        pass


def getCropIndecies3D(obj, framexb, frameyb, framezb):
    frameb = pre.np.zeros(6, int)
    framec = pre.np.zeros(6, int)
    
    xc  = obj.xc
    yc  = obj.yc
    zc  = obj.zc
    xb  = obj.xb
    yb  = obj.yb
    zb  = obj.zb
    
    switchx = False
    switchy = False
    switchz = False
    # Note that the +1's are added to make sure xc is contained in xb and
    # xc etc are symmetric, as in xc = [-a,a]
    if pre.np.sum(pre.np.abs(framexb)) > 0:
        idx0 = (pre.np.abs(xb-framexb[0])).argmin()
        idx1 = (pre.np.abs(xb-framexb[-1])).argmin()
        frameb[0:2] = [idx0, idx1+1]
        switchx = True
    else:
        frameb[0:2] = [0, xb.shape[0]+1]
    
    if pre.np.sum(pre.np.abs(frameyb)) > 0:
        idx0 = (pre.np.abs(yb-frameyb[0])).argmin()
        idx1 = (pre.np.abs(yb-frameyb[-1])).argmin()
        frameb[2:4] = [idx0, idx1+1]
        switchy = True
    else:
        frameb[2:4] = [0, yb.shape[0]+1]
    
    if pre.np.sum(pre.np.abs(framezb)) > 0:
        idx0 = (pre.np.abs(zb-framezb[0])).argmin()
        idx1 = (pre.np.abs(zb-framezb[-1])).argmin()
        frameb[4:] = [idx0, idx1+1]
        switchz = True
    else:
        frameb[4:] = [0, zb.shape[0]+1]
    
    if switchx:
        xb = xb[frameb[0]:frameb[1]]
        xc1 = 0.5 * (xb[1]  + xb[0])
        xc2 = 0.5 * (xb[-1] + xb[-2])
        idx0 = (pre.np.abs(xc-xc1)).argmin()
        idx1 = (pre.np.abs(xc-xc2)).argmin()
        framec[0:2] = [idx0, idx1+1]
        xc = xc[framec[0]:framec[1]]
    else:
        framec[0:2] = [0,xc.shape[0]+1]
    
    if switchy:
        yb = yb[frameb[2]:frameb[3]]
        yc1 = 0.5 * (yb[1]  + yb[0])
        yc2 = 0.5 * (yb[-1] + yb[-2])
        idx0 = (pre.np.abs(yc-yc1)).argmin()
        idx1 = (pre.np.abs(yc-yc2)).argmin()
        framec[2:4] = [idx0, idx1+1]
        yc = yc[framec[2]:framec[3]]
    else:
        framec[2:4] = [0,yc.shape[0]+1]
    
    if switchz:
        zb = zb[frameb[4]:frameb[5]]
        zc1 = 0.5 * (zb[1]  + zb[0])
        zc2 = 0.5 * (zb[-1] + zb[-2])
        idx0 = (pre.np.abs(zc-zc1)).argmin()
        idx1 = (pre.np.abs(zc-zc2)).argmin()
        framec[4:] = [idx0, idx1+1]
        zc = zc[framec[4]:framec[5]]
    else:
        framec[4:] = [0,zc.shape[0]+1]
    
    return xb, yb, zb, xc, yc, zc, frameb, framec

def get_datname_lare(datnum):
    if   len(str(datnum)) == 1: out = '000'+ str(datnum) + '.sdf'
    elif len(str(datnum)) == 2: out = '00'+ str(datnum)  + '.sdf'
    elif len(str(datnum)) == 3: out = '0'+ str(datnum)   + '.sdf'
    elif len(str(datnum)) == 4: out = str(datnum)        + '.sdf'
    return out
