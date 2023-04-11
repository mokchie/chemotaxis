import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
import matplotlib
from swimmer import Conc_field
matplotlib.rcParams.update({'font.size':14, 'font.family':'sans-serif'})
cmap = cm.get_cmap('jet')
fig1 = plt.figure()
ax1 = plt.axes(projection='3d')
#sname = 'testone-n2-swinging'
sname = 'test-DDQN-n4-xi0.0-DRL'
pattern = re.compile(sname+"-epoch-([0-9]+).data$")
filenames = []
epochs = []
direct = "data"
for root, dirs, files in os.walk(direct):
    if root == direct:
        for name in files:
            found = pattern.match(name)
            if found:
                epochs.append(int(found.groups()[0]))
                filenames.append(name)
                print(name)

files = sorted(zip(epochs, filenames))
conc_field = Conc_field(c0=20,k=1)
Gain  = []
Epoch = []
for epch, filename in files:
    X = []
    Y = []
    Z = []
    with open(direct+'/'+filename) as fp:
        for line in fp:
            t, rx, ry, rz, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            X.append(rx)
            Y.append(ry)
            Z.append(rz)
    if epch % 1 == 0:
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
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
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax1.plot([xb], [yb], [zb], 'w')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

plt.show()