import numpy as np

def getCellVals3D(data,quantity):
    '''
        Here we perform 2-point-per-direction interpolations of various raw quantities.
        This is the standard interpolation scheme used in Lare.
    '''
    if "v" in quantity:
        out = 0.125 * (data + np.roll(data, -1, axis=0) + np.roll(data, -1, axis=1) + np.roll(data, -1, axis=(0,1)) + 
        np.roll(data, -1, axis=2) + np.roll(data, -1, axis=(0,2)) + np.roll(data, -1, axis=(1,2)) + np.roll(data, -1, axis=(0,1,2)))  
        out = out[0:-1,0:-1,0:-1]
    elif quantity == "bx":
        out = 0.5*(data + np.roll(data,-1,axis=0))
        out = out[0:-1,:,:]    
    elif quantity == "by":
        out = 0.5*(data + np.roll(data,-1,axis=1))
        out = out[:,0:-1,:]    
    elif quantity == "bz":
        out = 0.5*(data + np.roll(data,-1,axis=2))
        out = out[:,:,0:-1]
    else:
        out = np.nan
        print("Enter an appropriate quantitiy! Returning NaN.")
    
    return out