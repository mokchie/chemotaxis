import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re,os
import numpy as np
from swimmer import Conc_field
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
conc_field = Conc_field(c0=20,k=1).get_conc
cmap = cm.get_cmap('jet')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
filenames = ['test1-sw-3d-epoch-1.data','test2-sw-3d-epoch-1.data','test3-sw-3d-epoch-1.data']
labels = ['2','4','8']
colors = ['C0','C1','C2']
direct = "data"
k0 = 4.0
tau0 = 4.0
v0 = 1
dt = 0.02
TA = [1/2,1/4,1/8]
for j,filename in enumerate(filenames):
    Taction = TA[j]
    ntimes = int(Taction * 2 * np.pi / k0 / v0 / dt)
    X = []
    Y = []
    Z = []
    Time = []
    with open(direct+'/'+filename) as fp:
        for line in fp:
            t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            print(tx*nx+ty*ny+tz*nz,tx*bx+ty*by+tz*bz,bx*nx+by*ny+bz*nz)
            X.append(x)
            Y.append(y)
            Z.append(z)
            Time.append(t)
    ax.plot(X, Y, Z, '-', color=colors[j], label=labels[j], linewidth=1)

plt.show()
