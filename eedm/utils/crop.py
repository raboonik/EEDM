import numpy as np

def getCropIndecies3D_lare(obj, framexb, frameyb, framezb):
    frameb = np.zeros(6, int)
    framec = np.zeros(6, int)
    
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
    if np.sum(np.abs(framexb)) > 0:
        idx0 = (np.abs(xb-framexb[0])).argmin()
        idx1 = (np.abs(xb-framexb[-1])).argmin()
        frameb[0:2] = [idx0, idx1+1]
        switchx = True
    else:
        frameb[0:2] = [0, xb.shape[0]+1]
    
    if np.sum(np.abs(frameyb)) > 0:
        idx0 = (np.abs(yb-frameyb[0])).argmin()
        idx1 = (np.abs(yb-frameyb[-1])).argmin()
        frameb[2:4] = [idx0, idx1+1]
        switchy = True
    else:
        frameb[2:4] = [0, yb.shape[0]+1]
    
    if np.sum(np.abs(framezb)) > 0:
        idx0 = (np.abs(zb-framezb[0])).argmin()
        idx1 = (np.abs(zb-framezb[-1])).argmin()
        frameb[4:] = [idx0, idx1+1]
        switchz = True
    else:
        frameb[4:] = [0, zb.shape[0]+1]
    
    if switchx:
        xb = xb[frameb[0]:frameb[1]]
        xc1 = 0.5 * (xb[1]  + xb[0])
        xc2 = 0.5 * (xb[-1] + xb[-2])
        idx0 = (np.abs(xc-xc1)).argmin()
        idx1 = (np.abs(xc-xc2)).argmin()
        framec[0:2] = [idx0, idx1+1]
        xc = xc[framec[0]:framec[1]]
    else:
        framec[0:2] = [0,xc.shape[0]+1]
    
    if switchy:
        yb = yb[frameb[2]:frameb[3]]
        yc1 = 0.5 * (yb[1]  + yb[0])
        yc2 = 0.5 * (yb[-1] + yb[-2])
        idx0 = (np.abs(yc-yc1)).argmin()
        idx1 = (np.abs(yc-yc2)).argmin()
        framec[2:4] = [idx0, idx1+1]
        yc = yc[framec[2]:framec[3]]
    else:
        framec[2:4] = [0,yc.shape[0]+1]
    
    if switchz:
        zb = zb[frameb[4]:frameb[5]]
        zc1 = 0.5 * (zb[1]  + zb[0])
        zc2 = 0.5 * (zb[-1] + zb[-2])
        idx0 = (np.abs(zc-zc1)).argmin()
        idx1 = (np.abs(zc-zc2)).argmin()
        framec[4:] = [idx0, idx1+1]
        zc = zc[framec[4]:framec[5]]
    else:
        framec[4:] = [0,zc.shape[0]+1]
    
    return xb, yb, zb, frameb, xc, yc, zc, framec


def getCropIndecies3D_cc(obj, framexc, frameyc, framezc):
    framec = np.zeros(6, int)
    
    xc  = obj.xc
    yc  = obj.yc
    zc  = obj.zc
    
    if np.sum(np.abs(framexc)) > 0:
        idx0 = (np.abs(xc-framexc[0]) ).argmin()
        idx1 = (np.abs(xc-framexc[-1])).argmin()
        framec[0:2] = [idx0, idx1]
    else:
        framec[0:2] = [0, xc.shape[0]]
    
    if np.sum(np.abs(frameyc)) > 0:
        idx0 = (np.abs(yc-frameyc[0]) ).argmin()
        idx1 = (np.abs(yc-frameyc[-1])).argmin()
        framec[2:4] = [idx0, idx1]
    else:
        framec[2:4] = [0, yc.shape[0]]
    
    if np.sum(np.abs(framezc)) > 0:
        idx0 = (np.abs(zc-framezc[0]) ).argmin()
        idx1 = (np.abs(zc-framezc[-1])).argmin()
        framec[4:6] = [idx0, idx1]
    else:
        framec[4:6] = [0, xc.shape[0]]
    
    return xc, yc, zc, framec
    
    