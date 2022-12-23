import numpy as np
from numpy import sqrt
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
def vel_field(x,y,z,alpha,check=False):
    r2 = x**2+y**2+z**2
    r = sqrt(r2)
    ri5 = 1/(r2*r2*r)
    ri7 = ri5/r2
    xy = x*y
    vx = 2*y - y*ri5 - 5*xy*x*(ri5-ri7)
    vy = -x*ri5 - 5*xy*y*(ri5-ri7)
    vz = -5*xy*z*(ri5-ri7)
    if check:
        mask = r<1
        vx[mask] = 0
        vy[mask] = 0
        vz[mask] = 0
    return vx*alpha/2,vy*alpha/2,vz*alpha/2

def get_interp(filename):
    with open(filename,'rb') as fp:
    t = np.load(fp)
    print(t)
    X = np.transpose(np.load(fp),(1,0,2))
    Y = np.transpose(np.load(fp),(1,0,2))
    Z = np.transpose(np.load(fp),(1,0,2))
    U = np.transpose(np.load(fp),(1,0,2))
    V = np.transpose(np.load(fp),(1,0,2))
    W = np.transpose(np.load(fp),(1,0,2))
    C = np.transpose(np.load(fp),(1,0,2))
    print('interpolating ....')
    interp = RegularGridInterpolator((X[:,0,0], Y[0,:,0], Z[0,0,:]),C,bounds_error=False,method='cubic')
    print('finish')
    return interp

class Conc_field:
    def __init__(self,interp):
        self.interp = interp
        self.k = np.abs(interp(1,0,0)-interp(50,0,0))/50
    def get_conc(self,x,y,z=0):
        return interp(x,y,z)

