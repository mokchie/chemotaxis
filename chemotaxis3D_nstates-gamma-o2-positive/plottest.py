import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
import matplotlib
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from swimmer_v2 import *
matplotlib.rcParams.update({'font.size':14, 'font.family':'sans-serif'})
cmap = cm.get_cmap('jet')
fig1 = plt.figure()
ax1 = plt.axes(projection='3d')
fig2,ax2 = plt.subplots(1,1)
pattern = re.compile("test-DDQN-n4-xi0.08-DRL-epoch-([0-9]+).data$")
filenames = []
epochs = []
direct = "data"
conc_field = Conc_field(c0=200,k=1)
for root, dirs, files in os.walk(direct):
    if root == direct:
        for name in files:
            found = pattern.match(name)
            if found:
                epochs.append(int(found.groups()[0]))
                filenames.append(name)
                print(name)

files = sorted(zip(epochs, filenames))
Gain  = []
Epoch = []
for epch, filename in files:
    X = []
    Y = []
    Z = []
    Xr = []
    Yr = []
    Zr = []    
    with open(direct+'/'+filename) as fp:
        for line in fp:
            t, rx, ry, rz, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            r0 = kappa/(kappa**2+tau**2)                        
            X.append(rx)
            Y.append(ry)
            Z.append(rz)
            Xr.append(rx+nx*r0)
            Yr.append(ry+ny*r0)
            Zr.append(rz+nz*r0)            
    if epch % 4 == 0:
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        Xr = np.array(Xr)
        Yr = np.array(Yr)
        Zr = np.array(Zr)
        x0 = 0#X[0]
        y0 = 0#Y[0]
        z0 = 0#Z[0]
        X = X-x0
        Y = Y-y0
        Z = Z-z0
        ax1.plot3D(X, Y, Z, '-', color=cmap(epch / np.max(epochs)), linewidth=1)
        ax1.scatter((X[-1],),(Y[-1],),(Z[-1],),s=10,c='k')
        Epoch.append(epch)
        Gain.append(conc_field.get_conc(X[-1],Y[-1],Z[-1])-conc_field.get_conc(X[0],Y[0],Z[0]))
ax2.plot(Epoch,Gain)
ax2.plot(Epoch,np.zeros_like(Epoch)+np.average(Gain),'k--')
ax2.set_xlabel('N')
ax2.set_ylabel(r'$\Delta c$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.show()
